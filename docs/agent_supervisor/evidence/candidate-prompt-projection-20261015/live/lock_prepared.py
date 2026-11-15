"""Bind prepare output before the single authorized execute step."""
from pathlib import Path
import hashlib,json
A=Path(__file__).resolve().parent

def bind(path):
 raw=path.read_bytes();return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

r=json.loads((A/'recipe.json').read_text());out=Path(r['output'])
assert json.loads((A/'prepare-result.json').read_text())['exit_code']==0
assert json.loads((A/'verify-result.json').read_text())['exit_code']==0
prep=json.loads((out/'preparation.json').read_text())
assert prep['prepared'] is True and prep['config_sha256']==bind(out/'config.json')['sha256']
with (A/'prepared-lock.json').open('x') as f:
 json.dump(dict(schema='public-validation-prepared-lock@1',files=[bind(out/'config.json'),bind(out/'preparation.json')],
 archive_verification=bind(A/'verify-result.json'),source_freeze=bind(A/'source-freeze.json'),provider_calls=0),f,indent=2,sort_keys=True)
 f.write('\n')
print(json.dumps(dict(status='prepared_locked',provider_calls=0)))
