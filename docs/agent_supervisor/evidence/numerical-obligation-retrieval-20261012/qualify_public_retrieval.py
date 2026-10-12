"""Root-gated, provider-free public-source numerical obligation qualification.

Uses actual numerical reports and native DuckLakes. Does not run the target,
models, providers, solvers, Docker, hidden verifier, or any solution source.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import socket, subprocess, sys, time, traceback

W=Path('/home/barberb/lift_coding')
R=W/'artifacts/numerical-obligation-retrieval-20261012'
P=W/'.worktrees/supervisor-comparison-contracts-20261011'
D=W/'.worktrees/numerical-obligation-datasets-20261012'
OLD=W/'artifacts/supervisor-ranked-successor-20261012/full-coding-comparison/numerical-public-probe-01'
SOURCE=W/'.benchmarks/terminal-bench-2/largest-eigenval/environment/src'
HASHES={'eigen.py':'bbd5be97fb11e9ed46fd5f1fb9d407efb19b265b549d3c36e2cfba923efe3d9d','eval.py':'70a12a80b9caa392b1f1f0439de853932c5ac61f1a2274b5de1ce93aebd4e139'}
LIMIT=4
BUDGET=8192

def wire(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(wire(v)).hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def bindings(paths):return {name:sha(name) for name in paths}

def configure(output):
 old=W/'artifacts/supervisor-ranked-successor-20261012/current-comparison-archive-05/host-execution-environment.json'
 env=dict(json.loads(old.read_bytes())['environment_overrides'])
 paths=env['PYTHONPATH'].split(':');paths[1]=str(D);env['PYTHONPATH']=':'.join(paths)
 env.update(IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(output/'orchestration'),IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(output/'seal.duckdb'))
 os.environ.update(env);os.environ.pop('IPFS_ACCELERATE_META_INDEX_DUCKDB',None);sys.path[:0]=paths
 def forbid(*a,**k):raise RuntimeError('network forbidden in offline obligation qualification')
 socket.socket.connect=forbid;socket.create_connection=forbid
 return env

def reports_for(receipt):
 out=Path(receipt['output']);index=json.loads((out/'reports.json').read_bytes());reports={}
 for path in sorted(HASHES):
  ref=index[path];file=out/ref['artifact'];assert sha(file)==ref['sha256'];reports[path]=json.loads(file.read_bytes())
 assert set(index)==set(HASHES)
 return reports

def pages(native,repository,receipt,path,expected,*,function_id=None,kind=None):
 cursor=None;seen=[];observations=[];actual_pages=[]
 for _ in range(100):
  page=native.retrieve_numerical_obligations(repository=repository,expected_receipt=receipt,path=path,cursor=cursor,function_id=function_id,kind=kind,limit=LIMIT,max_bytes=BUDGET)
  raw=wire(page);assert len(raw)<=BUDGET and len(page['rows'])<=LIMIT
  assert page['schema']=='supervisor-numerical-obligation-page@1'
  assert page['dataset_schema']=='numerical-obligation-page@1'
  assert page['context_receipt_sha256']==receipt['receipt_sha256']
  assert page['source_tree_id']==receipt['record']['source_tree_id']
  assert page['binding']['path']==path and page['binding']['source_sha256']==HASHES[path]
  assert page['matched_count']==len(expected) and page['offset']==len(seen)
  assert page['returned_count']==len(page['rows'])
  assert page['rows']==expected[len(seen):len(seen)+page['returned_count']]
  assert page['proof_authority'] is False and page['source_obligations_discharged']==0
  assert page['source_semantics_verified'] is False and page['whole_program_semantics_verified'] is False
  assert page['provider_calls']==page['model_inference_calls']==page['solver_calls']==0
  seen.extend(page['rows']);assert page['remaining_count']==len(expected)-len(seen)
  actual_pages.append(page);observations.append({'offset':page['offset'],'returned_count':page['returned_count'],'remaining_count':page['remaining_count'],'complete_response_utf8_bytes':len(raw),'response_sha256':digest(page)})
  cursor=page['next_cursor']
  if cursor is None:
   assert page['exhausted'] is True;break
  assert page['exhausted'] is False and page['returned_count']>0
  assert cursor['context_receipt_sha256']==receipt['receipt_sha256']
 else:raise AssertionError('bounded page loop did not finish')
 assert seen==expected
 return actual_pages,observations

def expect_rejected(label,call):
 try:call()
 except ValueError as e:return {'case':label,'rejected':True,'exception_type':type(e).__name__,'reason':str(e)[:300]}
 raise AssertionError(label+' was accepted')

def cold(output):
 configure(output)
 from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
 receipt=json.loads((output/'captured-context.json').read_bytes());repository=output/'public-inputs'
 assert native.validate_numerical_context(repository=repository,expected_receipt=receipt)==receipt
 reports=reports_for(receipt);observations={}
 for path,report in reports.items():
  actual,_=pages(native,repository,receipt,path,report['pending_obligations'])
  previous=json.loads((output/('pages-'+path+'.json')).read_bytes());assert actual==previous
  observations[path]={'page_count':len(actual),'pages_sha256':digest(actual),'rows':sum(x['returned_count'] for x in actual)}
 result={'schema':'numerical-obligation-cold-replay@1','exact_pages_replayed':True,'real_ducklake_verified':receipt['hydration']['ducklake_verified'],'hydration':receipt['hydration'],'per_file':observations,'provider_calls':0,'model_runs':0,'source_executed':False}
 write(output/'cold-replay.json',result)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cold',action='store_true');args=parser.parse_args()
 release=json.loads((R/'root-release.json').read_bytes())
 assert release['run_provider_free_qualification'] is True
 assert release['runner_sha256']==sha(__file__)
 assert bindings(release['source_bindings'])==release['source_bindings']
 output=args.output.absolute();assert output.parent==R and output.name.startswith('qualification-')
 if args.cold:cold(output);return 0
 output.mkdir(mode=0o700);start=time.monotonic();outcome={'schema':'numerical-public-obligation-qualification@1','complete':False,'provider_calls':0,'model_runs':0,'solver_runs':0,'docker_runs':0,'source_executed':False,'hidden_test_or_solution_source_read':False,'native_provider_tokens':None,'paid_token_savings_measured':False,'source_obligations_discharged':0,'source_semantics_verified':False,'whole_program_semantics_verified':False}
 try:
  env=configure(output);write(output/'command.json',{'argv':sys.argv,'interpreter':sys.executable,'runner_sha256':sha(__file__),'source_before':release['source_bindings'],'environment_overrides':env,'public_source_hashes':HASHES,'root_release_sha256':sha(R/'root-release.json')})
  repository=output/'public-inputs';repository.mkdir()
  for path,expected in HASHES.items():assert sha(SOURCE/path)==expected;(repository/path).write_bytes((SOURCE/path).read_bytes())
  from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
  assert Path(native.__file__).resolve()==P/'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py'
  receipt=native.prepare_numerical_context(repository=repository,source_hashes=HASHES,output=output/'native-context',profile=native.PROFILE)
  write(output/'captured-context.json',receipt);assert receipt['hydration']['ducklake_verified'] is True
  assert native.validate_numerical_context(repository=repository,expected_receipt=receipt)==receipt
  reports=reports_for(receipt);expected_count=sum(len(v['pending_obligations']) for v in reports.values());assert expected_count==45
  old=json.loads((OLD/'native-context/planning-summary.json').read_bytes());assert old['pending_obligation_count']==old['omitted_obligation_summaries']==45
  summary=receipt['summary'];assert summary['pending_obligation_count']==45
  kinds=dict(sorted(Counter(row['kind'] for report in reports.values() for row in report['pending_obligations']).items()));assert summary['pending_obligation_kinds']==kinds
  ref=summary['obligation_retrieval'];assert ref['requires_validated_context_receipt'] is True and ref['max_page_bytes']==8192 and ref['max_page_rows']==8
  assert ref['source_paths']==sorted(HASHES) and ref['omitted_source_paths']==0
  per_file={};all_keys=[];first_pages={}
  for path,report in reports.items():
   expected=report['pending_obligations'];actual,obs=pages(native,repository,receipt,path,expected)
   for page in actual:
    assert page['binding']['report_id']==report['report_sha256'] and page['binding']['report_sha256']==digest(report)
    assert page['total_count']==len(expected) and page['filtered_out_count']==0
   write(output/('pages-'+path+'.json'),actual);first_pages[path]=actual[0]
   all_keys.extend((path,x['obligation_id']) for page in actual for x in page['rows'])
   filters=[]
   for fid in sorted({fid for row in expected for fid in row['function_ids']}):
    selected=[row for row in expected if fid in row['function_ids']];filtered,fobs=pages(native,repository,receipt,path,selected,function_id=fid)
    filters.append({'function_id':fid,'matched_rows':len(selected),'pages':len(filtered),'page_digest':digest(filtered)})
   kind=sorted({row['kind'] for row in expected})[0]
   selected=[row for row in expected if row['kind']==kind];filtered,_=pages(native,repository,receipt,path,selected,kind=kind)
   filters.append({'kind':kind,'matched_rows':len(selected),'pages':len(filtered)})
   fid=expected[0]['function_ids'][0];selected=[row for row in expected if fid in row['function_ids'] and row['kind']==kind]
   filtered,_=pages(native,repository,receipt,path,selected,function_id=fid,kind=kind);filters.append({'function_id':fid,'kind':kind,'matched_rows':len(selected),'pages':len(filtered)})
   empty,_=pages(native,repository,receipt,path,[],function_id='function:missing');assert empty[0]['returned_count']==0
   per_file[path]={'expected_rows':len(expected),'retrieved_rows':sum(x['returned_count'] for x in actual),'page_count':len(actual),'page_observations':obs,'filter_checks':filters,'unknown_function_empty':True,'full_report_utf8_bytes':len(wire(report)),'retrieved_complete_page_utf8_bytes':sum(len(wire(x)) for x in actual)}
  assert len(all_keys)==len(set(all_keys))==45
  negatives=[]
  call=lambda **kw:native.retrieve_numerical_obligations(repository=repository,expected_receipt=kw.pop('expected_receipt',receipt),path=kw.pop('path','eigen.py'),limit=LIMIT,max_bytes=BUDGET,**kw)
  original=(repository/'eigen.py').read_bytes()
  try:
   (repository/'eigen.py').write_bytes(original+b'\n# authored stale source\n');negatives.append(expect_rejected('edited_source',lambda:call()))
  finally:(repository/'eigen.py').write_bytes(original)
  forged=deepcopy(receipt);forged['receipt_sha256']='0'*64;negatives.append(expect_rejected('changed_receipt',lambda:call(expected_receipt=forged)))
  cursor=deepcopy(first_pages['eigen.py']['next_cursor']);assert cursor is not None;cursor['context_receipt_sha256']='0'*64;negatives.append(expect_rejected('foreign_context_cursor',lambda:call(cursor=cursor)))
  cursor=deepcopy(first_pages['eigen.py']['next_cursor']);negatives.append(expect_rejected('changed_cursor_filter',lambda:call(cursor=cursor,kind='matrix_shape')))
  negatives.append(expect_rejected('foreign_path_cursor',lambda:call(path='eval.py',cursor=cursor)))
  assert native.validate_numerical_context(repository=repository,expected_receipt=receipt)==receipt
  result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(output),'--cold'],env={**os.environ,**env},capture_output=True,text=True,timeout=180)
  (output/'cold.stdout').write_text(result.stdout);(output/'cold.stderr').write_text(result.stderr);assert result.returncode==0,'cold replay failed; retained diagnostic'
  outcome.update(exact_unique_obligations_retrieved=45,previously_omitted_obligations_retrieved=45,per_file=per_file,negative_checks=negatives,cold_replay=json.loads((output/'cold-replay.json').read_bytes()),hydration=receipt['hydration'],summary_bytes={'prior':len(wire(old)),'current':len(wire(summary))},full_report_bytes=sum(len(wire(v)) for v in reports.values()),retrieved_complete_page_bytes=sum(v['retrieved_complete_page_utf8_bytes'] for v in per_file.values()),max_complete_page_bytes=max(o['complete_response_utf8_bytes'] for v in per_file.values() for o in v['page_observations']),authority_preserved=True,complete=True)
 except Exception as error:
  outcome.update(error_type=type(error).__name__,error=str(error)[:500]);(output/'failure.txt').write_text(traceback.format_exc())
 finally:
  outcome['source_unchanged']=bindings(release['source_bindings'])==release['source_bindings'];outcome['original_public_inputs_unchanged']=all(sha(SOURCE/p)==v for p,v in HASHES.items());outcome['complete']=outcome['complete'] and outcome['source_unchanged'] and outcome['original_public_inputs_unchanged'];outcome['seconds']=time.monotonic()-start
  write(output/'qualification.json',outcome);print(json.dumps({k:outcome.get(k) for k in ('complete','seconds','error_type','source_unchanged','exact_unique_obligations_retrieved')}))
 return 0 if outcome['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
