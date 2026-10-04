from pathlib import Path
import hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as E
B=Path(__file__).resolve().parent;W=B.parent.parent
D=W/'.worktrees/ir-admission-observation-datasets-20261004'
kind=sys.argv[1]
recipe=json.loads((W/'artifacts/source384-admission-native-20261004/native_clean-tests-command.json').read_text())
env=recipe['environment_overrides'].copy()
env['IPFS_DATASETS_PUBLIC_BOTTLE_FIXTURE']='/home/barberb/lift_coding/artifacts/source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py'
for key,suffix in [('IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB','seal.duckdb'),('IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH','keys.db'),('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH','host-scheduler.json')]: env[key]=str(B/(kind+'-'+suffix))
if kind=='native': targets=['tests/integration/logic/software_contracts/test_codebase_source_units_384.py']
else:
 assert kind=='integration'
 targets=['tests/integration/logic/software_contracts/test_codebase_ir.py','tests/integration/logic/software_contracts/test_codebase_current.py','tests/integration/logic/software_contracts/test_codebase_header_context.py','tests/integration/logic/software_contracts/test_duckdb_ast_authority.py','tests/integration/logic/software_contracts/test_duckdb_ast_shadow.py','tests/unit/logic/software_contracts/test_codebase_manifest_memo.py','tests/unit/logic/software_contracts/test_codebase_ir_untrusted_previous.py','tests/unit/logic/software_contracts/test_duckdb_ast_store.py','tests/unit/logic/software_contracts/test_duckdb_ast_store_persistence.py','tests/unit/logic/software_contracts/test_duckdb_ast_batch_reads.py','tests/unit/logic/formalization/autoencoder/test_function_span_line_index.py']
owners=['ipfs_datasets_py/logic/software_contracts/codebase_ir.py','ipfs_datasets_py/logic/software_contracts/duckdb_ast_store.py','ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py','ipfs_datasets_py/logic/formalization/autoencoder/source_function_units.py','ipfs_datasets_py/logic/formalization/autoencoder/security/security_formalization_evaluation.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={str(D/p):sha(D/p) for p in owners+targets}
argv=[recipe['argv'][0],'-B','-m','pytest','-q',*targets,'--junitxml='+str(B/(kind+'.xml'))]
def write(name,data):
 with (B/name).open('x') as f:json.dump(data,f,indent=2)
write(kind+'-command.json',dict(argv=argv,cwd=str(D),environment_overrides=env,source_pins=pins))
started=time.monotonic()
with (B/(kind+'.stdout')).open('x') as out,(B/(kind+'.stderr')).open('x') as err:
 run=subprocess.run(argv,cwd=D,env={**os.environ,**env},stdout=out,stderr=err,timeout=600)
tree=E.parse(B/(kind+'.xml'))
result=dict(returncode=run.returncode,seconds=time.monotonic()-started,tests=len(tree.findall('.//testcase')),failed=len(tree.findall('.//failure')),errors=len(tree.findall('.//error')),skipped=len(tree.findall('.//skipped')),source_pins_unchanged=all(sha(Path(p))==v for p,v in pins.items()))
write(kind+'-exit.json',result);print(json.dumps(result),flush=True)
assert not run.returncode and result['source_pins_unchanged'] and not result['skipped']
