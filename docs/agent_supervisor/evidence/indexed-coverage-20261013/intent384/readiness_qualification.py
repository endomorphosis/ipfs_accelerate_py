"""Current preparation readiness against exact existing local assets; no inference."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import time

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_intent_action_readiness as readiness
from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_checkpoint_384 as checkpoint
from ipfs_datasets_py.logic.formalization.autoencoder import source_embeddings_384 as embeddings
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured

R = Path(__file__).resolve().parent
def sha(raw): return hashlib.sha256(raw).hexdigest()
def save(name,value):
    path=R/name
    assert not path.exists()
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return dict(path=str(path),sha256=sha(path.read_bytes()),bytes=path.stat().st_size)

counts=Counter()
codes={checkpoint.load_intent_action_checkpoint_384.__code__:'checkpoint_load_calls',
       embeddings.embed_texts.__code__:'embedding_calls',structured.Runtime.infer.__code__:'numerical_inference_calls'}
def profile(frame,event,arg):
    if event=='call' and frame.f_code in codes: counts[codes[frame.f_code]]+=1

config=deployment.load_intent_action_384_config(R/'selected-config.json')
started=time.monotonic()
sys.setprofile(profile)
try:
    assets,binding=deployment._intent_action_384_assets(config)
    manifest={'intent_action_384':binding,'files':[dict(path=path,sha256=sha(raw),bytes=len(raw)) for path,raw in assets]}
    del assets
    manifest_binding=save('readiness-checked-asset-inventory.json',manifest)
    result={'schema':'intent384-real-asset-readiness-qualification@1','source_sha256':sha(Path(readiness.__file__).read_bytes()),
        'asset_inventory':manifest_binding,'inventory_from_checked_local_asset_bytes':True,
        'archive_created_or_verified':False,'cases':[], 'provider_calls':0,'training_steps':0,'downloads':0}
    controls=[('controlled_scalar','the calculator must compute result; requires left > 0; ensures result = old(right) + old(left) and returned.'),
              ('largest-eigenval_original_instruction',Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/instruction.md').read_text())]
    for label,instruction in controls:
        for selection,selected_config in [('explicit_local_config',config),('binding_only',None)]:
            before=counts.copy()
            observed=readiness.intent_action_readiness(instruction,manifest=manifest,config=selected_config)
            actual={key:counts[key]-before[key] for key in codes.values()}
            assert actual['embedding_calls']==actual['numerical_inference_calls']==0
            assert actual['checkpoint_load_calls']==int(selected_config is not None)
            assert observed['checkpoint_load_eligibility']==('eligible' if selected_config else 'not_checked')
            assert observed['input_scope']==('in_scope' if label=='controlled_scalar' else 'out_of_scope')
            result['cases'].append(dict(label=label,selection=selection,observation=observed,actual_call_counts=actual))
    before=counts.copy()
    unselected=readiness.intent_action_readiness(controls[1][1],manifest={})
    assert counts==before and unselected['selection']=='not_selected'
    result['unselected']=unselected
    result['expectations_met']=True
finally:
    sys.setprofile(None)
result['actual_call_counts']={key:counts[key] for key in codes.values()}
result['seconds']=time.monotonic()-started
result['limitations']=['Binding-only observations validate the actual local asset inventory structure; no archive was created or read in this qualification.',
    'Eligibility validates the current shared checkpoint loader, not successful numerical decode, formal projection, instruction meaning or proof.']
record=save('readiness-qualification-result.json',result)
print(json.dumps({'result':record,'cases':[dict(label=x['label'],selection=x['selection'],readiness=x['observation']['inference_readiness']) for x in result['cases']], 'counts':result['actual_call_counts']},sort_keys=True))
