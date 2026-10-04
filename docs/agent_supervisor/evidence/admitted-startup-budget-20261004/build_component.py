from pathlib import Path
from xml.etree import ElementTree as E
import copy,hashlib,json,shutil
B=Path(__file__).resolve().parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
P=B/'public-evidence';P.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(P/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def read(name):return json.loads((B/name).read_text())
def cases(label):
 root=E.parse(B/(label+'.xml')).getroot();rows=[]
 for c in root.findall('.//testcase'):
  status=next((k for k in ('failure','error','skipped') if c.find(k) is not None),'passed')
  row={'id':c.attrib.get('classname','')+'::'+c.attrib['name'],'status':status}
  if status=='failure':
   msg=c.find('failure').attrib.get('message','')
   row['reason']=('missing_supervisor_api' if 'has no attribute' in msg else 'unix_socket_path_too_long' if 'AF_UNIX path too long' in msg else 'startup_health_deadline' if ('sustained health' in msg or 'test_health_rejects_stale' in row['id']) else 'unclassified')
  rows.append(row)
 return rows
labels=['baseline-01','bounds-01','native-01','native-02','native-03','baseline-api-01']
history={}
for label in labels:
 rows=cases(label);history[label]={'exit':read(label+'-exit.json'),'counts':{k:sum(r['status']==k for r in rows) for k in ('passed','failure','error','skipped')},'raw_xml_sha256':sha(B/(label+'.xml')),'cases':rows}
 for suffix in ['-command.json','-exit.json']:shutil.copyfile(B/(label+suffix),P/(label+suffix))
 if label in ('baseline-01','native-03'):
  shutil.copyfile(B/(label+'.xml'),P/(label+'.xml'))
 elif label in ('native-02','baseline-api-01'):
  root=E.parse(B/(label+'.xml')).getroot()
  for c,row in zip(root.findall('.//testcase'),rows):
   for child in list(c):
    if child.tag in ('system-out','system-err'):c.remove(child)
    elif child.tag in ('failure','error','skipped'):
     child.text='[Body omitted from public export]'
     child.attrib={'message':row.get('reason',child.tag)}
  E.ElementTree(root).write(P/(label+'-bounded.xml'),encoding='utf-8',xml_declaration=True)
passed02={r['id'] for r in history['native-02']['cases'] if r['status']=='passed'}
passed03={r['id'] for r in history['native-03']['cases'] if r['status']=='passed'}
failed02={r['id'] for r in history['native-02']['cases'] if r['status']=='failure'}
known={r['id'] for r in history['baseline-api-01']['cases'] if r['status']=='failure'}
assert len(passed02|passed03)==84 and failed02-passed03==known and len(known)==5
assert history['native-03']['exit']['returncode']==0
assert all(history[label]['exit']['source_pins_unchanged'] for label in ['native-02','native-03','baseline-api-01'])
for label in ['native-02','native-03']:
 assert all(sha(A/f)==digest for f,digest in read(label+'-command.json')['source_pins'].items())
observation={'schema':'admitted-startup-budget-component@1','distinct_current_passing_controls':84,'known_unresolved_legacy_controls':sorted(known),'skips':0,'accounted_prior_isolated_failure':sorted(failed02&passed03),'source_pins_stable':True,'current_controls':['native-02','native-03'],'original_owner_reproduction':'baseline-api-01','prior_generation_not_qualifying':['baseline-01','bounds-01','native-01'],'native_01_limitation':'Production/test pins changed during this superseded run; long fixture Unix paths and two startup deadlines also failed.','full_suite_green_claimed':False,'benchmark_result_claimed':False,'original_trial_startup_subcause_proved':False,'production_checks_removed':False,'global_catalog_changed':False,'raw_production_source_bodies_exported':False,'private_stores_exported':False,'failed_xml_note':'Bounded XML keeps every case and status/count while omitting failure bodies, arbitrary messages and captured streams; original raw XML digests are retained.'}
write('observation.json',observation);write('control-history.json',history)
original={str(p.relative_to(B/'original')):sha(p) for p in (B/'original').rglob('*.py')}
write('original-source-pins.json',original)
write('baseline-overlay-bindings.json',{'schema':'original-owner-import-overlay@1','finder_sha256':sha(B/'original-overlay/sitecustomize.py'),'selected_modules':['ipfs_accelerate_py.agent_supervisor.control.control_plane','ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime','ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime','ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor'],'unchanged_portal_class_sha256':original['ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py'],'available_original_files':original,'source_bodies_exported':False})
shutil.copyfile(B/'original-overlay/sitecustomize.py',P/'original-overlay-finder.py')
for name in ['run_controls.py','native-02-runner.py','native-03-runner.py','build_component.py']:shutil.copyfile(B/name,P/name)
delay=Path('/tmp/ipfs-asb-native-02/test_signed_extended_start_all0/launch/startup-control-observation.json');shutil.copyfile(delay,P/'authored-delay-observation.json')
prior=Path('/home/barberb/lift_coding/artifacts/extended-supervisor-checker-20261004/native-startup-failure-detail.json');shutil.copyfile(prior,P/'prior-native-failure.json')
(P/'README.md').write_text('''# Signed native startup budget

The current affected runtime scope has **84 distinct passing controls, zero skips, and five known unresolved legacy bootstrap tests**. `native-02` ran 89 cases: 83 passed and six failed. `native-03` reran the eight isolated-runtime cases using a fresh private orchestration directory; all eight passed, correcting the one remaining isolated failure without changing production. Their union counts overlapping cases once. The five remaining tests expect methods absent from `PortalImplementationSupervisor`; `baseline-api-01` reproduced all five failures with original owner modules and the byte-identical committed supervisor class. This is not a claim that the entire selected suite is green.

The earlier official full trial reached a verified local-contract Doctor candidate, then START refused because sustained health was not proved within 20 seconds; STOP recorded an active lifecycle transaction conflict. That generation retained no per-boundary validation timings. Its exact startup subcause remains unknown. Neither this test reproduction nor a later resource sample establishes that old subcause.

The authored baseline reproduces the same health refusal by adding 20.1 seconds of prelaunch work to a real native START. The current native control retains every validation check and adds 20.1 seconds during launch validation plus 31 seconds during bootstrap validation. Actual observed checks took 20.147 and 31.085 seconds. With the explicitly signed local START and bootstrap allowances of 120 seconds, real START and STOP succeeded, one exact child bootstrap receipt was issued, no bootstrap errors occurred, and all child processes were reaped. These delays are controlled test inputs, not measured benchmark work or evidence of host-pressure recovery.

Only an explicitly selected local benchmark launch may carry the START override. It is capped at 120 seconds, bound to the signed manifest, inherited bootstrap environment and local control-service allowance, and constrained by the launch lifetime. The default control catalog and discovery remain unchanged at 30 seconds; normal runtime request limits remain unchanged. The benchmark driver retains STOP at 20 seconds and the separate cleanup reserve. Source/currentness, signatures, exact process birth, grants, health, and proof gates still run. Tests reject unsigned or mutated runtime, service, manifest and bootstrap-environment budgets and preserve all non-timeout and STOP bounds.

Bounded diagnostics report fresh control, launch and bootstrap validation durations and completion status without source, credentials or error bodies. The related driver/profile component is separate; its counts are not added here. No new Docker qualification, full benchmark success, token score, whole-program proof, or performance advantage is claimed by this host package.

`native-01` is retained as a superseded, nonqualifying run: source/test pins drifted, six legacy socket fixtures exceeded AF_UNIX path bounds, and two empty-runtime startup cases failed. A short fresh `/tmp` basetemp fixed the socket issue; an explicitly isolated orchestration environment fixed the remaining empty-runtime startup case. No production adjustment was made to satisfy those fixture failures. Original raw XML digests are retained. Passing baseline/native-03 XML is exported verbatim; mixed/failed XML is explicitly bounded to case metadata and outcomes so traceback source bodies and captured streams are excluded. Commands, exact selected source/test digests, exit records, original-owner import mapping and artifact recipes are retained. Source snapshots, private stores and process logs are excluded. Selected pins are not a complete import inventory; the next archive qualification owns that broader binding.
''')
files=[]
for p in sorted(P.iterdir()):
 if p.is_file():files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
write('manifest.json',{'schema':'admitted-startup-budget-public-evidence@1','files':files})
print(json.dumps({'path':str(P),'members':len(files),'member_bytes':sum(x['bytes'] for x in files),'manifest_sha256':sha(P/'manifest.json')}))
