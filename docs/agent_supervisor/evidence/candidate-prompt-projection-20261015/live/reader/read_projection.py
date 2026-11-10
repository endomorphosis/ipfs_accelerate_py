"""Add closed candidate representation observations without changing prior metrics."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 256_000
MODE = 'native-context-object@1'
FLAGS = frozenset({'proof_authority','execution_authority','completion_authority','omission_authority',
    'source_freshness_verified','program_semantics_proved','provider_token_usage_verified',
    'complete_run_token_savings_claimed','raw_content_exported'})
VIEW_NAMES = ('original_input','native_prompt','native_context','native_suffix','public_instruction_block','candidate_view')
COMPLETE_NAMES = ('native_complete','candidate_complete','selected_complete')
PAIR_SUFFIXES = ('_sha256','_bytes')
VIEW_KEYS = {'schema','view_schema','source_schema','representation_lossless','evidence_bodies_preserved',
    'unknown_fields_preserved',*FLAGS,*(n+s for n in VIEW_NAMES for s in PAIR_SUFFIXES)}
SELECTION_KEYS = {'schema','requested_mode','selected_mode','fallback_reason','selection_basis','view_receipt',
    'view_receipt_sha256',*FLAGS,*(n+s for n in COMPLETE_NAMES for s in PAIR_SUFFIXES)}


def need(condition):
    if not condition:
        raise ValueError('invalid_closed_candidate_projection')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    raw=Path(path).read_bytes()
    return dict(path=str(path),sha256=digest(raw),bytes=len(raw))


def pairs(value,names):
    for name in names:
        d,n=value[name+'_sha256'],value[name+'_bytes']
        need(type(d) is str and len(d)==64 and all(c in '0123456789abcdef' for c in d))
        need(type(n) is int and 0<=n<=MAX_BYTES)


def validate_selection(value):
    need(type(value) is dict and set(value)==SELECTION_KEYS)
    need(value['schema']=='supervisor-candidate-prompt-selection@1' and value['requested_mode']==MODE
        and value['selection_basis']=='complete_utf8_bytes' and all(value[k] is False for k in FLAGS))
    v=value['view_receipt']
    need(type(v) is dict and set(v)==VIEW_KEYS)
    need(v['schema']=='supervisor-candidate-prompt-projection@1'
        and v['view_schema']=='supervisor-candidate-provider-view@1'
        and type(v['source_schema']) is str and v['source_schema'] in {
            'supervisor-owner-candidate-router-input@1','supervisor-owner-candidate-router-input@2',
            'supervisor-owner-candidate-router-input@3'}
        and all(v[k] is True for k in ('representation_lossless','evidence_bodies_preserved','unknown_fields_preserved'))
        and all(v[k] is False for k in FLAGS))
    pairs(v,VIEW_NAMES);pairs(value,COMPLETE_NAMES)
    need(v['native_prompt_bytes']==v['native_context_bytes']+v['native_suffix_bytes'])
    need(value['view_receipt_sha256']==digest(canonical(v)))
    need(value['native_complete_bytes']>=v['original_input_bytes']
        and value['candidate_complete_bytes']>=v['candidate_view_bytes']
        and value['native_complete_bytes']-v['original_input_bytes']==value['candidate_complete_bytes']-v['candidate_view_bytes'])
    smaller=value['candidate_complete_bytes']<value['native_complete_bytes']
    need(value['selected_mode']==(MODE if smaller else 'legacy')
        and value['fallback_reason']==(None if smaller else 'complete_input_not_smaller'))
    chosen='candidate_complete' if smaller else 'native_complete'
    need(all(value['selected_complete'+s]==value[chosen+s] for s in PAIR_SUFFIXES))
    need(len(canonical(value))<=MAX_BYTES)
    return value


def project_receipt(receipt):
    need(type(receipt) is dict and receipt.get('schema')=='router-implementation-invocation@1'
        and receipt.get('purpose')=='coding')
    if 'candidate_prompt_projection' not in receipt:
        return dict(status='absent',independent_prompt_replay=False)
    s=validate_selection(receipt['candidate_prompt_projection']);v=s['view_receipt']
    dispatch=receipt.get('candidate_round_dispatch')
    need(type(dispatch) is dict and dispatch.get('schema') in {
        'supervisor-candidate-dispatch-observation@1','supervisor-candidate-dispatch-observation@2'})
    for source,target in (('provider_input','prompt'),('native_prompt','native_prompt')):
        need(all(type(dispatch.get(source+k)) is type(receipt.get(target+k))
            and dispatch.get(source+k)==receipt.get(target+k) for k in PAIR_SUFFIXES))
    for source,target in (('original_input','prompt'),('native_prompt','native_prompt')):
        need(all(type(v[source+k]) is type(receipt.get(target+k)) and v[source+k]==receipt[target+k] for k in PAIR_SUFFIXES))
    need(all(type(s['selected_complete'+k]) is type(receipt.get('model_prompt'+k))
        and s['selected_complete'+k]==receipt['model_prompt'+k] for k in PAIR_SUFFIXES))
    return dict(status='closed_schema_arithmetic_and_receipt_joins_verified',
        independent_prompt_replay=False,current_source_verified=False,token_measurement=False,
        schema=s['schema'],requested_mode=s['requested_mode'],selected_mode=s['selected_mode'],
        fallback_reason=s['fallback_reason'],selection_basis=s['selection_basis'],
        **{n+k:s[n+k] for n in COMPLETE_NAMES for k in PAIR_SUFFIXES},
        complete_input_bytes_reduced=s['native_complete_bytes']-s['selected_complete_bytes'],
        selection_receipt_sha256=digest(canonical(s)),view_receipt_sha256=s['view_receipt_sha256'],
        native_representation_lossless_reported=v['representation_lossless'],
        evidence_bodies_preserved_reported=v['evidence_bodies_preserved'],
        unknown_fields_preserved_reported=v['unknown_fields_preserved'],
        **{key:False for key in FLAGS})


def project_report(report):
    need(type(report) is dict and type(report.get('provider_invocations')) is list
        and len(report['provider_invocations'])<=64)
    observations=[];seen={};duplicates=0
    for r in report['provider_invocations']:
        need(type(r) is dict)
        if r.get('purpose')!='coding':continue
        identity=r.get('invocation_id')
        need(type(identity) is str and 1<=len(identity)<=256)
        fingerprint=digest(canonical(r))
        if identity in seen:
            need(seen[identity]==fingerprint);duplicates+=1;continue
        seen[identity]=fingerprint
        observations.append(project_receipt(r))
    return dict(status='closed_candidate_observations',coding_receipts=len(observations),
        duplicate_receipts_ignored=duplicates,observations=observations,
        independent_owner_replay_performed=False,raw_content_exported=False,
        provider_calls_by_reader=0,total_token_savings_established=False)


def baseline_reader():
    frozen=json.loads((ROOT/'upstream-freeze.json').read_text())
    for row in frozen['files']:need(binding(row['path'])==row)
    entry=Path(frozen['entrypoint'])
    spec=importlib.util.spec_from_file_location('_projection_baseline_planning',entry)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def read_run(directory):
    # Read only existing retained result reports through the same bounded pure reader.
    baseline=baseline_reader().baseline_reader();bindings=[];rows=[]
    for path in sorted(Path(directory).glob('jobs/supervisor-full-largest-eigenval/largest-eigenval__*/agent/supervisor-result.json')):
        report=baseline.read(path,bindings)
        try: observation=project_report(report)
        except (ValueError,TypeError,KeyError,OverflowError,RecursionError):
            observation=dict(status='invalid_closed_candidate_projection',independent_owner_replay_performed=False)
        rows.append(dict(trial=path.parent.parent.name,observation=observation))
    return dict(status='observed' if rows else 'no_retained_trial',trials=rows,artifact_bindings=bindings)


def comparison(recipe,prior=None):
    base=baseline_reader().read_comparison(recipe,prior_directory=prior)
    r=json.loads(Path(recipe).read_text())
    return dict(schema='candidate-projection-additive-readout@1',baseline=base,
        candidate_projection=dict(pilot=read_run(r['output']),prior=read_run(prior) if prior else None),
        additive_reader_sha256=digest(Path(__file__).read_bytes()),
        raw_prompts_or_provider_text_exported=False,provider_calls_by_reader=0,
        independent_owner_replay_performed=False,total_token_savings_established=False,
        causal_savings_claimed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--prior',type=Path)
    args=p.parse_args();print(json.dumps(comparison(args.recipe,args.prior),indent=2,sort_keys=True,allow_nan=False))
