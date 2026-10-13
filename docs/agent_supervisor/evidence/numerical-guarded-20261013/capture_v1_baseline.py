"""Freeze admitted public-source v1 results before adding a selected grammar."""
from pathlib import Path
import hashlib
import json

from ipfs_datasets_py.logic.software_verification import numerical_source, numerical_logic
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384

ROOT = Path('/home/barberb/lift_coding')
OUT = Path(__file__).resolve().parent / 'baseline-v1.json'
PUBLIC = ROOT / '.benchmarks/terminal-bench-2/largest-eigenval/environment/src'

def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def sha(value):
    return hashlib.sha256(value).hexdigest()

rows = []
for name in ('eigen.py', 'eval.py'):
    raw = (PUBLIC / name).read_bytes()
    report = numerical_logic.formalize_numerical_source(raw.decode(), path=name)
    assert report == numerical_logic.validate_numerical_logic_report(report, raw.decode(), path=name)
    rows.append(dict(path=name, source_sha256=sha(raw), source_bytes=len(raw),
        report_sha256=report['report_sha256'], canonical_report_sha256=sha(wire(report)),
        per_function=report['summary']['per_function'],
        pending_obligations=len(report['pending_obligations']),
        lean_source_sha256=report['projections']['lean']['source_sha256'],
        source_obligations_discharged=0))
result = dict(schema='numerical-guarded-v1-baseline@1', files=rows,
    producer_sha256={m.__name__:sha(Path(m.__file__).read_bytes()) for m in (numerical_source, numerical_logic)},
    structured_checkpoint_implementation=structured_source_384._implementation(),
    original_source_executed=False, provider_calls=0, model_inference_calls=0,
    hidden_verifier_read=False, lake_executed=False)
assert not OUT.exists()
OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(path=str(OUT), sha256=sha(OUT.read_bytes()),
    pending_obligations=sum(x['pending_obligations'] for x in rows))))
