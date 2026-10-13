"""Fresh local selected-Intent qualification; no downloads, training or LLM."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter

R = Path(__file__).resolve().parent
P = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
D = Path('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013')
from ipfs_accelerate_py.agent_supervisor.runtime import intent_384_advisor as advisor
from ipfs_accelerate_py.agent_supervisor.runtime import intent_advisor_selection as selection
from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_runtime_384 as owner
from ipfs_datasets_py.logic.formalization.autoencoder import source_embeddings_384 as embeddings
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured
from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_checkpoint_384 as checkpoint_owner
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment

def sha(raw): return hashlib.sha256(raw).hexdigest()
def wire(value): return json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
def save(name,value):
    path=R/name
    assert not path.exists()
    path.write_bytes(wire(value))
    return {'path':str(path),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}

counts=Counter()
codes={checkpoint_owner.load_intent_action_checkpoint_384.__code__:'intent_checkpoint_load_calls',
       structured.Runtime.infer.__code__:'structured_inference_calls',
       embeddings.embed_texts.__code__:'gte_embedding_calls'}
def profile(frame,event,arg):
    if event != 'call': return
    code=frame.f_code
    if code in codes: counts[codes[code]]+=1
    if code.co_name=='__init__' and code.co_filename.endswith('/sentence_transformers/SentenceTransformer.py'):
        counts['gte_model_initializations']+=1

config_path=R/'selected-config.json'
config=deployment.load_intent_action_384_config(config_path)
weights=Path(config['checkpoint_path'])
before=sha(weights.read_bytes())
assert before == config['checkpoint_sha256']
assert Path(advisor.__file__).is_relative_to(P) and Path(owner.__file__).is_relative_to(D)
report={'schema':'selected-intent384-local-qualification@1','configuration_path':str(config_path),
        'configuration_sha256':sha(config_path.read_bytes()),'checkpoint_sha256':before,
        'producer_pins':{str(Path(module.__file__)) :sha(Path(module.__file__).read_bytes())
                         for module in (advisor,selection,owner,embeddings,structured,checkpoint_owner,deployment)},
        'versions':{name:importlib.metadata.version(name) for name in ('torch','sentence-transformers','transformers','tokenizers','numpy')},
        'training_steps':0,'provider_calls':0,'download_calls':0,'default_model_promoted':False,
        'source_semantics_verified':False,'proof_authority':False,'cases':[]}
started=time.monotonic()
try:
    sys.setprofile(profile)
    try:
        assets,binding=deployment._intent_action_384_assets(config)
        report['packaging_readiness']={'status':'ready','asset_count':len(assets),'asset_bytes':sum(len(raw) for _,raw in assets),
            'binding':binding,'model_inference_executed':False,'archive_built':False}
        del assets
    except Exception as error:
        report['packaging_readiness']={'status':'unavailable','error_type':type(error).__name__,'reason':str(error)[:512],
                                     'model_inference_executed':False,'archive_built':False}
    instruction_path=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/instruction.md')
    benchmark_instruction=instruction_path.read_text()
    controls=[('controlled_scalar','the calculator must compute result; requires left > 0; ensures result = old(right) + old(left) and returned.'),
              ('largest-eigenval_original_instruction',benchmark_instruction)]
    for label,instruction in controls:
        initial=counts.copy();case_started=time.monotonic()
        advice,selected,elapsed=selection.prepare_intent_384_selection(instruction=instruction,config_path=config_path)
        path=R/(label+'-advice.json');saved=save(path.name,advice)
        summary,checked=selection.load_intent_384_planner_summary(path=path,expected_sha256=saved['sha256'],
            instruction=instruction,selection=selected)
        native=advice.get('report') or {}
        replay=checked.get('report') or {}
        report['cases'].append({'label':label,'instruction_sha256':sha(instruction.encode()),'instruction_bytes':len(instruction.encode()),
            'status':advice['status'],'shared_status':native.get('status'),'reason':native.get('reason'),'checkpoint_compatibility':native.get('checkpoint_compatibility'),
            'error_type':advice.get('error_type'),'failure_stage':advice.get('failure_stage'),
            'shared_model_inference_executed':native.get('model_inference_executed',False),
            'shared_embedding_provenance_present':native.get('embedding_provenance') is not None,
            'shared_producer_pins_present':native.get('producer_pins') is not None,
            'numerical_replay_verified':advice['numerical_replay_verified'],
            'saved_advice_replay_status':checked['status'],'saved_advice_shared_status':replay.get('status'),
            'saved_advice_replay_equal':advice==checked,
            'planner_summary_present':summary is not None,'planner_summary_bytes':len(summary.encode()) if summary else 0,
            'planner_summary_sha256':sha(summary.encode()) if summary else None,
            'source_preserved':advice['instruction_sha256']==sha(instruction.encode()) and advice['raw_instruction_preserved'],
            'actual_call_counts':{key:counts[key]-initial[key] for key in set(counts)|set(initial)},
            'preparation_seconds':elapsed/1e9,'total_seconds':time.monotonic()-case_started,
            'advice':saved,'formal_logic_projection_executed':False,'lake_executed':False,
            'proof_authority':False,'completion_authority':False,'execution_authority':False})
    report['case_expectations_met']=(report['cases'][0]['status']=='semantic_candidate_advice'
        and report['cases'][0]['shared_model_inference_executed'] is True
        and report['cases'][0]['numerical_replay_verified'] is True
        and report['cases'][0]['planner_summary_present'] is True
        and report['cases'][1]['status']=='fail_open_input_out_of_scope'
        and report['cases'][1]['shared_model_inference_executed'] is False
        and report['cases'][1]['planner_summary_present'] is False)
finally:
    sys.setprofile(None)
    report['actual_call_counts']=dict(counts)
    report['seconds']=time.monotonic()-started
    report['checkpoint_bytes_unchanged']=sha(weights.read_bytes())==before
    report['embedding_models_in_private_cache']=len(embeddings._MODEL_CACHE)
    report['limitations']=['This is local optional preprocessing, not a supervisor benchmark or task correctness test.',
        'The supported control is authored controlled language, not arbitrary benchmark instruction semantics.',
        'The source-audited action route has no separate projection request and performs no Lake build.',
        'The unsupported benchmark instruction is refused before checkpoint numerical inference; selecting these weights cannot broaden their grammar.',
        'Function-call instrumentation counts loaded shared functions without replacing them; timings include profiling and replay costs.']
    save('qualification-result.json',report)
print(json.dumps({'result':str(R/'qualification-result.json'),'packaging':report['packaging_readiness']['status'],
                  'cases':[{'label':x['label'],'status':x['status'],'inference':x['shared_model_inference_executed'],
                            'counts':x['actual_call_counts']} for x in report['cases']],
                  'expectations_met':report.get('case_expectations_met')},sort_keys=True))
