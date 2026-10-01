"""Optional consumer boundaries; actual learned kernels are tested in datasets."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_unit_advice as api


INSTRUCTION = "The maintainer must validate input.\n"


@pytest.fixture
def native(monkeypatch):
    calls = []
    report = {"schema":"source-document-autoencoder/v1", "source_sha256":api._sha(INSTRUCTION.encode()),
        "report_sha256":"1"*64, "candidates":[{"domain":"intent_ir"}],
        "counts":{"intent_candidates":1}, "security_regions":[],
        "intent":{"units":[{"accepted":True,"start_char":0,"end_char":len(INSTRUCTION),
            "inference":{"learned":{"frame":{"actor":"maintainer","action":"validate","object":"input","modality":"required"}}}}]}}
    def produce(instruction, intent, security):
        calls.append((instruction,intent,security))
        return deepcopy(report)
    monkeypatch.setattr(api, '_native', produce)
    return calls, report


def persist(tmp_path, advice):
    path = tmp_path/'source-unit-advice.json'
    raw = api._wire(advice)
    path.write_bytes(raw)
    return path, api._sha(raw)


def test_disabled_default_does_not_touch_native_model_or_descriptor(tmp_path, native):
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION,
        intent_descriptor_path=tmp_path/'missing.json')
    assert value['status'] == 'disabled' and native[0] == []
    assert value['raw_instruction_preserved'] and value['continue_planning']


def test_sidecar_pins_exact_instruction_and_replays_once(tmp_path, native):
    descriptor = tmp_path/'descriptor.json';descriptor.write_text('{"schema":"fixture"}')
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True,
        intent_descriptor_path=descriptor)
    path, digest = persist(tmp_path, value)
    loaded, costs = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded == value and len(native[0]) == 2
    assert costs['inference_replays'] == 1 and costs['replay_seconds'] >= 0
    assert loaded['descriptor_pins']['intent']['sha256'] == hashlib.sha256(descriptor.read_bytes()).hexdigest()
    summary = api.source_unit_planner_summary(loaded, maximum_bytes=8192)
    assert '"actor":"maintainer"' in summary
    assert 'no proof' in summary and '"execution_authority":false' in summary


@pytest.mark.parametrize('change', ['artifact', 'instruction', 'descriptor', 'report'])
def test_changed_evidence_is_omitted_and_raw_task_can_continue(tmp_path, native, change):
    descriptor = tmp_path/'descriptor.json';descriptor.write_text('{"schema":"fixture"}')
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True,
        intent_descriptor_path=descriptor)
    path, digest = persist(tmp_path, value)
    instruction = INSTRUCTION
    if change == 'artifact': path.write_bytes(path.read_bytes()+b' ')
    elif change == 'instruction': instruction += 'Changed.'
    elif change == 'descriptor': descriptor.write_text('{"schema":"changed"}')
    else: native[1]['candidates'] = []
    loaded, costs = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=instruction)
    assert loaded['status'] == 'fail_open_invalid_sidecar'
    assert loaded['continue_planning'] and loaded['raw_instruction_preserved']
    assert api.source_unit_planner_summary(loaded, maximum_bytes=8192) is None
    assert loaded['instruction_sha256'] == api._sha(instruction.encode())
    assert costs['inference_replays'] == (1 if change == 'report' else 0)


def test_rehashed_candidate_injection_still_fails_native_replay(tmp_path, native):
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True)
    value['report']['intent']['units'][0]['inference']['learned']['frame']['action'] = 'delete'
    value['advice_sha256'] = api._sha(api._wire({k:v for k,v in value.items() if k!='advice_sha256'}))
    path,digest = persist(tmp_path,value)
    loaded,costs = api.load_source_unit_advice(path=path,expected_sha256=digest,instruction=INSTRUCTION)
    assert loaded['status'] == 'fail_open_invalid_sidecar' and costs['inference_replays'] == 1


@pytest.mark.parametrize('failure', ['native', 'report_bound', 'missing_descriptor'])
def test_optional_stage_errors_never_replace_raw_instruction(tmp_path, monkeypatch, native, failure):
    options = {}
    if failure == 'native':
        def failed(*args): raise RuntimeError('unavailable native dependency')
        monkeypatch.setattr(api,'_native',failed)
    elif failure == 'report_bound': monkeypatch.setattr(api,'MAX_ADVICE_BYTES',10)
    else: options['security_descriptor_path'] = tmp_path/'missing.json'
    result = api.prepare_source_unit_advice(instruction=INSTRUCTION,enabled=True,**options)
    assert result['status'] == 'fail_open_error'
    assert result['report'] is None and result['instruction_sha256'] == api._sha(INSTRUCTION.encode())
    assert result['continue_planning'] and result['raw_instruction_preserved']


def test_summary_budget_is_not_relaxed(native):
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION,enabled=True)
    with pytest.raises(ValueError,match='existing planner bound'):
        api.source_unit_planner_summary(value, maximum_bytes=10)


def test_typed_security_projection_profiles_reach_bounded_advisory_summary(native):
    _, report = native
    report['security_regions'] = [{'equations':[{'unit_id':'unit:conditional', 'projection':{
        'status':'candidate', 'candidate_model':{'result':{'op':'if','sort':'Int'}},
        'typed_ir':{'schema':'security-pure-function-ir/v2'},
        'assumptions':['Exact built-in integer inputs.'], 'projection_sha256':'a'*64,
        'projections':[{'family_id':'program','profile_id':'program_ir'},
                       {'family_id':'smt','profile_id':'scalar_first_order_model_equality'}]}}]}]
    advice = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True)
    summary = api.source_unit_planner_summary(advice, maximum_bytes=8192)
    assert '"typed_ir_available":true' in summary
    assert '"family":"smt"' in summary and '"projection_sha256":"'+'a'*64+'"' in summary
    assert '"execution_authority":false' in summary


@pytest.fixture
def family_native(monkeypatch, native):
    calls, report = native
    report['intent_family_projection'] = {'report_sha256':'f'*64,
        'family_inventory':[{'family_id':'dcec','status':'available_views'},
                            {'family_id':'hyperproperty','status':'unsupported'}],
        'units':[{'unit_id':'unit:1', 'slot_environment':{'environment_sha256':'e'*64,
            'status':'ready','slots':[{'slot_id':'actor','resolved_sort':'Person',
                'origin':'knowledge_graph','referent_resolved':True}]},
            'typed_fixture':{'formula':{'op':'predicate','predicate':'validate','arguments':[{'slot_id':'actor'}]}}}]}
    def produce(instruction, intent, security, family_request=None):
        calls.append((instruction,intent,security,deepcopy(family_request)))
        return deepcopy(report)
    monkeypatch.setattr(api, '_native', produce)
    return calls, report


def test_explicit_family_request_replays_pinned_context_and_reaches_summary(tmp_path, family_native):
    context = tmp_path/'context.json'; context.write_text('{"schema":"fixture:context"}')
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True,
        project_logic_families=True, intent_family_context_path=context,
        requested_intent_families=['dcec','higher_order'])
    assert value['schema'] == api.FAMILY_SCHEMA
    path, digest = persist(tmp_path, value)
    loaded, costs = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded == value and costs['inference_replays'] == 1
    assert len(family_native[0]) == 2 and family_native[0][0] == family_native[0][1]
    summary = api.source_unit_planner_summary(loaded, maximum_bytes=8192)
    assert '"available_families":["dcec"]' in summary
    assert '"sort":"Person"' in summary and '"origin":"knowledge_graph"' in summary
    assert 'no existence, facts, or task correctness established' in summary


def test_changed_family_context_fails_before_inference_and_preserves_task(tmp_path, family_native):
    context = tmp_path/'context.json'; context.write_text('{"schema":"fixture:context"}')
    value = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True,
        project_logic_families=True, intent_family_context_path=context)
    path, digest = persist(tmp_path, value)
    context.write_text('{"schema":"changed"}')
    loaded, costs = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded['status'] == 'fail_open_invalid_sidecar' and costs['inference_replays'] == 0
    assert loaded['raw_instruction_preserved'] and loaded['continue_planning']


@pytest.mark.parametrize('options', [
    {'project_logic_families':True}, {'enabled':True,'intent_family_context_path':'context.json'},
    {'requested_intent_families':['dcec']}])
def test_family_options_require_enabled_stages(options):
    with pytest.raises(ValueError):
        api.prepare_source_unit_advice(instruction=INSTRUCTION, **options)


def test_family_schema_cannot_hide_context_in_report_free_envelope(tmp_path):
    value = api._base(INSTRUCTION, 'disabled', family_request={'injected':True})
    path, digest = persist(tmp_path, value)
    loaded, costs = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded['status'] == 'fail_open_invalid_sidecar' and costs['inference_replays'] == 0
