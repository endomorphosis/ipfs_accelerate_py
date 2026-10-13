"""Freeze existing public reports and checkpoint pins before adding loop grammar."""
import hashlib,json
from pathlib import Path
from ipfs_datasets_py.logic.software_verification import numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,program
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
r=Path(__file__).resolve().parent
public=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
raw=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
sha=lambda v:hashlib.sha256(v).hexdigest()
rows=[]
for logic in (numerical_logic,numerical_logic_v2):
    for name in ('eigen.py','eval.py'):
        source=(public/name).read_bytes()
        report=logic.formalize_numerical_source(source.decode(),path=name)
        assert report==logic.validate_numerical_logic_report(report,source.decode(),path=name)
        rows.append(dict(profile=logic.PROFILE,path=name,source_sha256=sha(source),source_bytes=len(source),canonical_report_sha256=sha(raw(report)),report_sha256=report['report_sha256'],per_function=report['summary']['per_function'],pending_obligations=len(report['pending_obligations']),lean_source_sha256=report['projections']['lean']['source_sha256']))
record=dict(schema='numerical-loop-baseline@1',files=rows,producer_sha256={m.__name__:sha(Path(m.__file__).read_bytes()) for m in (numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,program)},structured_checkpoint_implementation=structured_source_384._implementation(),provider_calls=0,model_inference_calls=0,original_source_executed=False,hidden_verifier_read=False)
with (r/'baseline.json').open('x') as out:json.dump(record,out,indent=2,sort_keys=True);out.write('\n')
print(json.dumps(dict(baseline_sha256=sha((r/'baseline.json').read_bytes()),profiles=2,files=4)))
