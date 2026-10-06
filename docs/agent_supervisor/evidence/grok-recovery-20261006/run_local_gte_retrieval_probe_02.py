"""Bounded authored CPU inference through the real offline retrieval preflight."""
from pathlib import Path
import hashlib, json, os, subprocess, time
A=Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
R=Path('/home/barberb/lift_coding/.worktrees/grok-neural-retrieval-revision-20261006')
D=Path('/home/barberb/lift_coding/.worktrees/terminal-bounded-header-datasets-20261005')
SNAPSHOT=Path('/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd')
PRIVATE=A/'private-gte-retrieval-probe-02';PRIVATE.mkdir(mode=0o700)
repository=PRIVATE/'repository';repository.mkdir()
(repository/'example.py').write_text('def clean_name(name: str) -> str:\n    return name.strip()\n\ndef total(values):\n    return sum(values)\n')
child=PRIVATE/'probe.py'
child.write_text('''import json, resource, sys, time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(60,65))
from benchmarks.agent_supervisor.container_coding import learned_vector_preflight as learned
from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import config_for
from benchmarks.agent_supervisor.container_coding.terminal_retrieval_selection import selected_retrieval_revision, require_retrieval_revision
root=Path(sys.argv[1]); snapshot=Path(sys.argv[2]); revision=snapshot.name
manifest={"learned_requirements":["selected-local-model"],"model_snapshot_revision":revision}
selected=selected_retrieval_revision(manifest,"full")
config=config_for(root,root/'trial',root/'archive',"full",model_revision=selected)
assert require_retrieval_revision(manifest,"full",config['agents'][0]['kwargs']['model_revision'])==revision
measurement={}
original=learned._LocalRouterModel
class MeasuredModel(original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        measurement.update(parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()), parameter_dtypes=sorted({str(p.dtype) for p in self.model.parameters()}))
learned._LocalRouterModel=MeasuredModel
start=time.monotonic()
result=learned.qualify(root,root/'vectors',['example.py'],'clean name',snapshot,selected)
assert result['schema']=='native-local-learned-vector-qualification@1' and result['status']=='qualified'
assert result['model_revision']==revision and result['dimensions']==384 and result['symbols']==2
assert result['local_model_calls']>0 and result['configuration']['device']=='cpu'
record={"schema":"local-gte-retrieval-probe@1","status":"qualified","model_revision":revision,"dimensions":result['dimensions'],"symbols":result['symbols'],"learned_embeddings":True,"local_model_calls":result['local_model_calls'],"canary_status":result['canary'].get('status'),"ducklake_projected":result['ducklake']['status']=='projected',"persisted_index_reopened_and_equal":True,"native_ast_rows_replayed":result['native_fact_rows_replayed'],"device":"cpu","local_files_only":result['configuration']['local_files_only'],"trust_remote_code":result['configuration']['trust_remote_code'],"torch_or_cuda_training_performed":False,"training_steps":0,"model_download_calls":0,"llm_provider_calls":0,"seconds":time.monotonic()-start,"peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"hard_memory_bound_inferred":False,"result_sha256":__import__('hashlib').sha256((root/'vectors/result.json').read_bytes()).hexdigest(),"source_sha256":__import__('hashlib').sha256((root/'example.py').read_bytes()).hexdigest(),"model_manifest_sha256":__import__('hashlib').sha256((root/'vectors/model-manifest.json').read_bytes()).hexdigest(),"versions":result['versions'],"raw_corpus_or_model_outputs_exported":False,**measurement}
Path(sys.argv[3]).write_text(json.dumps(record,sort_keys=True,indent=2)+'\\n')
''')
source_paths=['benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py','benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py','benchmarks/agent_supervisor/container_coding/terminal_retrieval_selection.py','benchmarks/agent_supervisor/container_coding/test_terminal_retrieval_revision.py']
def snapshot():
 return {p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in source_paths}
before=snapshot();started=time.monotonic()
env={**os.environ,'PYTHONPATH':str(R)+':'+str(D),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
output=A/'qualification/local-gte-retrieval-probe-02.json'
with (PRIVATE/'process.log').open('xb') as log:
 result=subprocess.run(['/home/barberb/.local/bin/python','-B',str(child),str(repository),str(SNAPSHOT),str(output)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
record={'schema':'local-gte-retrieval-process-boundary@1','exit_code':result.returncode,'seconds':time.monotonic()-started,'cpu_limit_seconds':60,'wall_timeout_seconds':90,'source_sha256':before,'source_unchanged':before==snapshot(),'log_sha256':hashlib.sha256((PRIVATE/'process.log').read_bytes()).hexdigest(),'provider_calls':0,'raw_output_exported':False}
with (A/'qualification/local-gte-retrieval-process-02.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(record))
if result.returncode:raise SystemExit(result.returncode)
print(output.read_text())
