"""Public-source replay, source CFG coverage, unchanged old profiles and real Lake."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
from ipfs_datasets_py.logic.software_verification import numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,numerical_source_v3,numerical_logic_v3,program
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
from ipfs_datasets_py.logic.formalization.autoencoder.native_family_lake_v4 import _execute
R=Path(__file__).resolve().parent
PUBLIC=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
raw=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
sha=lambda value:hashlib.sha256(value).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
AUTHORED='''def scan(items, flag):
    for item in items:
        if flag:
            continue
        if item < 0:
            break
        assert item, f"zero item: {item}"
    else:
        return 0
    while flag:
        flag = False
    else:
        return 1
'''
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lake',required=True);args=parser.parse_args()
    output=args.output.absolute();assert output.parent==R and not output.exists();output.mkdir();start=time.monotonic()
    baseline=json.loads((R/'baseline.json').read_bytes())
    historical=(numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,program)
    assert baseline['producer_sha256']=={m.__name__:sha(Path(m.__file__).read_bytes()) for m in historical}
    assert structured_source_384._implementation()==baseline['structured_checkpoint_implementation']
    modules=(*historical,numerical_source_v3,numerical_logic_v3)
    pins={m.__name__:sha(Path(m.__file__).read_bytes()) for m in modules}
    for row in baseline['files']:
        source=(PUBLIC/row['path']).read_bytes();assert sha(source)==row['source_sha256']
        logic=numerical_logic if row['profile']==numerical_logic.PROFILE else numerical_logic_v2
        assert sha(raw(logic.formalize_numerical_source(source.decode(),path=row['path'])))==row['canonical_report_sha256']
    files=[]
    for name in ('eigen.py','eval.py'):
        source=(PUBLIC/name).read_bytes();report=numerical_logic_v3.formalize_numerical_source(source.decode(),path=name)
        assert report==numerical_logic_v3.validate_numerical_logic_report(report,source.decode(),path=name)
        assert report['coverage']['source_obligations_discharged']==0
        assert all(row['status']=='pending' and row['function_ids'] and row['evaluation_contexts'] and not row['evaluation_guard_verified'] for row in report['pending_obligations'])
        write(output/(name+'.report.json'),report)
        lean=report['projections']['lean'];assert sha(lean['source'].encode())==lean['source_sha256'];(output/(name+'.lean')).write_text(lean['source'])
        build=_execute(lean['source'],'NumericalObligations',args.lake,60);write(output/(name+'.lake.json'),build);assert build['status']=='passed' and build['backend_executed']
        damaged=deepcopy(report);damaged['pending_obligations'][0]['evaluation_contexts']=[]
        try:numerical_logic_v3.validate_numerical_logic_report(damaged,source.decode(),path=name)
        except ValueError:pass
        else:raise AssertionError('tampered occurrence accepted')
        structural=report['structural'];cfgs=[f['cfg'] for f in structural['program']['functions']]
        files.append(dict(path=name,source_sha256=sha(source),per_function=report['summary']['per_function'],pending_obligations=len(report['pending_obligations']),obligation_kinds=dict(Counter(row['kind'] for row in report['pending_obligations'])),control_region_count=len(structural['control_regions']),control_region_kinds=dict(Counter(row['kind'] for row in structural['control_regions'])),cfg_blocks=sum(len(c['blocks']) for c in cfgs),cfg_edges=sum(len(c['edges']) for c in cfgs),compound_contract_count=len(report['compound_contracts']),assertion_requirement_count=len(report['assertion_requirements']),canonical_report_sha256=sha(raw(report)),lean_source_sha256=lean['source_sha256'],real_lake_status=build['status'],tampered_context_refused=True,source_obligations_discharged=0))
        assert sha((PUBLIC/name).read_bytes())==files[-1]['source_sha256']
    report=numerical_logic_v3.formalize_numerical_source(AUTHORED,path='authored_control.py');write(output/'authored-control.report.json',report)
    assert report['summary']['structurally_complete_functions']==1
    lean=report['projections']['lean']['source'];build=_execute(lean,'NumericalObligations',args.lake,60);write(output/'authored-control.lake.json',build);assert build['status']=='passed'
    negative=_execute(lean+'\ntheorem unsupported_source_truth : False := by trivial\n','NumericalObligations',args.lake,60);write(output/'false-source-truth.lake.json',negative)
    assert negative['status']=='failed' and negative['backend_executed'] and not negative['timed_out'] and not negative['output_truncated']
    assert pins=={m.__name__:sha(Path(m.__file__).read_bytes()) for m in modules}
    result=dict(schema='numerical-loop-public-source-qualification@1',complete=True,profile=numerical_logic_v3.PROFILE,files=files,baseline_sha256=sha((R/'baseline.json').read_bytes()),producer_sha256=pins,structured_checkpoint_pins_unchanged=True,historical_reports_unchanged=True,authored_control_build_passed=True,false_source_truth_build_rejected=True,real_lake_builds=4,original_public_source_executed=False,hidden_verifier_read=False,official_terminal_bench_score=None,native_provider_tokens=None,provider_calls=0,model_inference_calls=0,source_obligations_discharged=0,source_semantics_verified=False,seconds=time.monotonic()-start)
    write(output/'qualification.json',result)
    print(json.dumps(dict(complete=True,pending_obligations=sum(x['pending_obligations'] for x in files),complete_functions=sum(x['structurally_complete'] for f in files for x in f['per_function']),real_lake_builds=4)))
if __name__=='__main__':main()
