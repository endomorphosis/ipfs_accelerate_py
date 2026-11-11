"""Recheck immutable launch assets and exact owned-container absence."""
import hashlib,importlib.util,json,subprocess,time
from pathlib import Path
A=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_tool_diagnostic_launcher',A/'live/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
recipe=m.check();frozen=m.read(A/'live/source-freeze.json')
identity=m.read(A/'live/private-attempt-01/container-identity.json')['container_id']
assert type(identity) is str and len(identity)==64 and all(c in '0123456789abcdef' for c in identity)
absence=[]
for _ in range(2):
 values=subprocess.check_output(['docker','container','ls','--all','--quiet','--no-trunc','--filter','id='+identity],text=True,timeout=15).splitlines()
 absence.append(identity not in values)
 time.sleep(.2)
record=dict(schema='tool-event-census-final-freeze-check@1',source_frozen_unchanged=True,
 runtime_commit=frozen['roots']['source']['commit'],source_files_rechecked=len(frozen['source_sha256']),
 frozen_script_bindings_rechecked=len(frozen['files']),frozen_asset_bindings_rechecked=len(frozen['inputs']),
 exact_container_absence_observations=absence,
 source_freeze_sha256=m.sha(A/'live/source-freeze.json'),
 diagnostic_sha256=m.sha(A/'live/actual-public-diagnostic-01.json'),
 official_verifier_executed=False,raw_provider_content_read=False)
assert absence==[True,True]
m.write(A/'final-freeze-check-01.json',record)
print(json.dumps(record,sort_keys=True))
