"""Synthetic structural join controls; no training/inference/checker qualification.

Actual native IntentIR, matcher and passive code features are used. Inert
weights/ranks and lexical/KG bodies are declared unit controls, not learned
runtime observations or executed formal proofs.
"""
from copy import deepcopy
import ast
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_training_join as join
from ipfs_accelerate_py.agent_supervisor.planning import intent_codebase_matching as matching
from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae


def digest(value, *, ascii=True):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


def source_units(raw):
    result=[]; offsets=[0]
    for line in raw.splitlines(keepends=True):offsets.append(offsets[-1]+len(line))
    for node in ast.parse(raw).body:
        start=offsets[node.lineno-1]+node.col_offset;end=offsets[node.end_lineno-1]+node.end_col_offset
        result.append({'symbol':node.name,'role':{'_hkey':'field_name','_hval':'field_value'}[node.name],
            'line':node.lineno,'end_line':node.end_lineno,
            'source_span':{'start_byte':start,'end_byte':end,'sha256':hashlib.sha256(raw[start:end]).hexdigest()},
            'source_ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
            'conversion_symbol':'touni','normalization_ops':['identity'],'guarded':True})
    return result


def reseal_artifacts(inputs):
    receipt=inputs['training_receipt'];learner=inputs['learner']
    receipt['features_sha256']=digest(inputs['features'])
    inputs['checkpoint']['feature_manifest_sha256']=receipt['features_sha256']
    receipt['checkpoint_sha256']=digest(inputs['checkpoint']);receipt['index_sha256']=digest(inputs['learned_index'])
    learner['checkpoint_sha256']=receipt['checkpoint_sha256'];learner['receipt_sha256']=digest(receipt)
    learner['ranks']=[{k:r[k] for k in ('row_id','path','symbol','line','reconstruction_error')} for r in inputs['learned_index']['ranks']]
    context=inputs['source_context']
    context['checkpoint']={'checkpoint_sha256':learner['checkpoint_sha256'],'receipt_sha256':learner['receipt_sha256'],
        'features_sha256':receipt['features_sha256'],'index_sha256':receipt['index_sha256']}
    match=inputs['match_result'];match['current_source_snapshot']['source_context_sha256']=digest(context,ascii=False)
    inputs['match_result']=matching.match_intent_codebase(intent_document=match['intent_document'],
        source_text=match['intent_source']['text'],source_identity=match['intent_source']['identity'],
        query=match['query'],evidence_rows=[],current_source_snapshot=match['current_source_snapshot'])
    for control in inputs['model_controls']:
        control.update(input_features_sha256=receipt['features_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],source_hashes=receipt['source_hashes'])
        if control['name']=='trained':control['ranking']=deepcopy(inputs['learned_index']['ranks'])
        elif control['name']=='shuffled_order':control['ranking']=list(reversed(deepcopy(inputs['learned_index']['ranks'])))


@pytest.fixture
def inputs():
    from ipfs_datasets_py.logic.intent_ir.schema import (IntentIRDocument,IntentKind,IntentModality,
        IntentStatement,NodeGrounding,ReviewStatus,SourceRef,SourceSpan,StatementKind)
    public='Please repair bottle; retain every existing guard. μ\n'
    raw=b'def _hkey(value):\n    if len(value) >= 1:\n        return value\n\ndef _hval(value):\n    if len(value) >= 1:\n        return value\n'
    source_hash=hashlib.sha256(raw).hexdigest();ledger={'bottle.py':source_hash}
    rows,unsupported=ae._features({'bottle.py':raw},['bottle.py'],1024)
    features={'schema':ae.FEATURE_SCHEMA,'domain':ae.DOMAIN,'features':list(ae.FEATURES),'source_hashes':ledger,
        'paths':['bottle.py'],'rows':rows,'unsupported':unsupported,
        'normalization':'log1p counts, per-function L2 normalization','source_text_retained':False}
    weights=[[[0.25,0.25] for _ in ae.FEATURES],[0.0,0.0],[[0.25 for _ in ae.FEATURES] for _ in range(2)],[0.0 for _ in ae.FEATURES]]
    checkpoint={'schema':ae.CHECKPOINT_SCHEMA,'domain':ae.DOMAIN,'projection_family':'code-ast-control-flow-contract-advisory@1',
        'architecture':'tanh-linear-autoencoder@1','input_width':len(ae.FEATURES),'latent_width':2,'weights':weights,
        'feature_manifest_sha256':'pending','implementation':{'scope':'synthetic unit artifact, not runtime'},
        'legal_ir_weights_loaded':False,'legal_ir_views_loaded':False,'tla_projection':{'status':'unsupported'}}
    ranks=sorted([{k:r[k] for k in ('row_id','path','symbol','line')}|{'reconstruction_error':0.01,'latent':[0.25,0.25]} for r in rows],key=lambda r:(-r['reconstruction_error'],r['row_id']))
    rank_by_id={r['row_id']:r for r in ranks}
    native_mean=sum(rank_by_id[r['row_id']]['reconstruction_error'] for r in rows)/len(ranks)
    index={'mean_reconstruction_error':native_mean,'ranks':ranks,'authority':'unverified_candidate_only',
        'ranking_role':'nomination_order_only','omission_authority':False,'formalization_authority':False,'proof_authority':False}
    metrics={'epochs':1,'after_reconstruction_loss':native_mean,'holdout_evaluated':False,'training_scope':'synthetic_structural_unit_control_no_optimizer'}
    receipt={'schema':'supervisor-code-autoencoder-training@1','domain':ae.DOMAIN,'repository':'/synthetic/repository',
        'output':'/synthetic/code-autoencoder','source_hashes':ledger,'paths':['bottle.py'],'source_count':1,'sample_count':len(rows),
        'metrics':metrics,'max_functions':1024,'authority':'unverified_candidate_only','proof_authority':False,
        'formalization_authority':False,'legal_state_mutated':False}
    learner={'schema':ae.SCHEMA,'domain':ae.DOMAIN,'repository':receipt['repository'],'output':receipt['output'],
        'metrics':metrics,'epochs_completed':1,'source_hashes':ledger,'source_count':1,'sample_count':len(rows),
        'authority':'unverified_candidate_only','proof_authority':False}
    context={'schema':'terminal-codebase-training-source-context@1','source_path':'bottle.py','source_sha256':source_hash,
        'source_bytes':len(raw),'model_domain':ae.DOMAIN,'training_source_hashes':ledger,'checkpoint':{}}
    text_hash=hashlib.sha256(public.encode()).hexdigest()
    ref=SourceRef(ref_id='original-public-source',source_uri='fixture:public.md',source_id=text_hash,
        source_revision=text_hash,content_sha256=text_hash,review_status=ReviewStatus.TRUSTED_FIXTURE,span=SourceSpan(0,len(public)))
    statement=IntentStatement(statement_id='repair-goal',kind=StatementKind.GOAL,modality=IntentModality.REQUIRED,
        normalized_text=public,source_ref_ids=(ref.ref_id,),predicate='repair',arguments=('agent','bottle'),
        confidence=0.0,grounding=NodeGrounding.GROUNDED,review_status=ReviewStatus.TRUSTED_FIXTURE)
    extra=IntentStatement(statement_id='retain-guards',kind=StatementKind.GUARD,modality=IntentModality.ASSERTED,
        normalized_text='Retain every existing guard.',source_ref_ids=(ref.ref_id,),predicate='retain',arguments=('guards',),
        confidence=0.0,grounding=NodeGrounding.INFERRED)
    document=IntentIRDocument(document_id='synthetic-native-control',title='Authored structural test focus',
        intent_kind=IntentKind.DECLARATIVE,sources=(ref,),statements=(statement,extra)).to_dict()
    identity={k:getattr(ref,k) for k in ('ref_id','source_uri','source_id','source_revision','content_sha256')}
    query={'schema':matching.QUERY_SCHEMA,'review_ref':'synthetic-reviewed-focus-not-prompt-interpretation',
        'statement':{'statement_id':'repair-goal','predicate':'repair','arguments':['agent','bottle']},
        'source_path':'bottle.py','symbols':['_hkey','_hval'],'property':'header_delimiter_rejection','polarity':'positive',
        'domain':matching.reviewed_header_matching_domain(),'semantic_alignment_verified':False}
    snapshot={'schema':'terminal-codebase-proof-source-snapshot@1','source_path':'bottle.py','source_sha256':source_hash,
        'source_bytes':len(raw),'source_unit_bindings':source_units(raw),'source_context_sha256':'0'*64,
        'environment_sha256':'1'*64,'environment_ref_sha256':'2'*64,'translation_sha256':'3'*64}
    match=matching.match_intent_codebase(intent_document=document,source_text=public,source_identity=identity,
        query=query,evidence_rows=[],current_source_snapshot=snapshot)
    controls=[]
    for name,policy in join._POLICIES.items():
        ranking=deepcopy(ranks) if name=='trained' else list(reversed(deepcopy(ranks))) if name=='shuffled_order' else []
        c={'name':name,'ranking':ranking,'input_features_sha256':'pending','source_hashes':ledger,
            'checkpoint_sha256':'pending','inference_policy':policy,'training_executed':False}
        if name=='zero_heads':
            c['control_weights_sha256']=digest(join._zero(weights))
            c['ranking']=[{k:r[k] for k in ('row_id','path','symbol','line')}|{'latent':[0.0,0.0],
                'reconstruction_error':sum(x*x for x in r['features'])/len(ae.FEATURES)} for r in rows]
        controls.append(c)
    value={'match_result':match,'source_records':[{'path':'bottle.py','source_sha256':source_hash,'bytes':len(raw),'source_text':raw.decode()},
        {'path':'.supervisor-instruction.md','source_sha256':text_hash,'bytes':len(public.encode()),'source_text':public}],
        'source_context':context,'learner':learner,'training_receipt':receipt,'checkpoint':checkpoint,
        'features':features,'learned_index':index,'fixed_candidates':{'lexical':{'scope':'synthetic passive unit candidates','hits':['_hkey','_hval']},
        'kg':[{'source':'bottle._hkey','target':'value','relation':'synthetic passive unit edge'}]},'model_controls':controls}
    reseal_artifacts(value)
    return deepcopy(value)


def test_exact_original_intent_source_model_and_rows_join_without_authority(inputs):
    pristine=deepcopy(inputs);receipt=join.build_terminal_intent_training_join(**inputs)
    assert receipt['schema']==join.SCHEMA and receipt['original_native_match']['status']=='unknown'
    assert receipt['original_native_match']['intent_source']['text']==inputs['source_records'][1]['source_text']
    assert len(receipt['source_unit_nominations'])==2 and len(receipt['residual_requirements'])==2
    assert {r['feature_row_id'] for r in receipt['source_unit_nominations']}=={r['row_id'] for r in inputs['features']['rows']}
    assert all(r['status']=='unresolved_software_behavior' for r in receipt['residual_requirements'])
    assert all(receipt[name] is False for name in join._AUTHORITY)
    assert receipt['trained_inference_replayed_here'] is False
    assert receipt['join_sha256']=='sha256:'+digest({k:v for k,v in receipt.items() if k!='join_sha256'})
    assert join.validate_terminal_intent_training_join(receipt,**inputs)==receipt
    receipt['original_native_match']['query']['symbols'].clear()
    assert inputs==pristine
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='entire advisory join'):
        join.validate_terminal_intent_training_join(receipt,**inputs)
    forged=join.build_terminal_intent_training_join(**inputs)
    forged['eligible']=True
    forged['join_sha256']='sha256:'+digest({k:v for k,v in forged.items() if k!='join_sha256'})
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='entire advisory join'):
        join.validate_terminal_intent_training_join(forged,**inputs)


def test_model_off_zero_and_shuffled_controls_preserve_fixed_candidates_and_all_residuals(inputs):
    receipt=join.build_terminal_intent_training_join(**inputs)
    outcomes={r['control']['name']:r for r in receipt['model_control_outcomes']}
    fixed={tuple(r['fixed_candidate_ids']) for r in outcomes.values()}
    assert len(fixed)==1 and len(next(iter(fixed)))==2
    assert outcomes['model_off']['learned_candidate_order']==[]
    assert outcomes['shuffled_order']['learned_candidate_order']==list(reversed(outcomes['trained']['learned_candidate_order']))
    assert all(r['residual_requirements']==receipt['residual_requirements'] for r in outcomes.values())
    assert all(r[name] is False for r in outcomes.values() for name in join._AUTHORITY)
    bad=deepcopy(inputs);bad['model_controls'][2]['ranking'][0]['latent'][0]=0.25
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='zero control'):
        join.build_terminal_intent_training_join(**bad)
    bad=deepcopy(inputs);bad['model_controls'][1]['ranking']=deepcopy(bad['learned_index']['ranks'])
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='model-off'):
        join.build_terminal_intent_training_join(**bad)


def test_resealed_same_feature_guard_and_symbol_row_reassignment_are_rejected(inputs):
    bad=deepcopy(inputs);raw=bad['source_records'][0]['source_text'].replace('>=','>').encode()
    newrows,_=ae._features({'bottle.py':raw},['bottle.py'],1024)
    assert [r['features'] for r in newrows]==[r['features'] for r in inputs['features']['rows']]
    assert [r['ast_sha256'] for r in newrows]!=[r['ast_sha256'] for r in inputs['features']['rows']]
    h=hashlib.sha256(raw).hexdigest();bad['source_records'][0].update(source_text=raw.decode(),source_sha256=h,bytes=len(raw))
    ledger={'bottle.py':h}
    for owner in (bad['learner'],bad['training_receipt'],bad['features']):owner['source_hashes']=ledger
    bad['source_context'].update(source_sha256=h,source_bytes=len(raw),training_source_hashes=ledger)
    bad['match_result']['current_source_snapshot'].update(source_sha256=h,source_bytes=len(raw),source_unit_bindings=source_units(raw))
    reseal_artifacts(bad)
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='current source AST replay'):
        join.build_terminal_intent_training_join(**bad)
    bad=deepcopy(inputs);a,b=bad['learned_index']['ranks']
    for field in ('path','symbol','line'):a[field],b[field]=b[field],a[field]
    reseal_artifacts(bad)
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='learned row identity reassigned'):
        join.build_terminal_intent_training_join(**bad)
    bad=deepcopy(inputs);bad['checkpoint']['weights'][0][0][0]+=0.1
    with pytest.raises(join.TerminalIntentTrainingJoinError,match='artifact or receipt digest'):
        join.build_terminal_intent_training_join(**bad)
