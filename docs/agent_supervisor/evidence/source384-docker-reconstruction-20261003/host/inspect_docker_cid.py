"""Optional, separate-process registry inspection before measured context starts."""
import json, pathlib, subprocess, time
B=pathlib.Path(__file__).parent
D=B/'docker-01'
assert (D/'deployment/python-install.log').is_file(), 'remaining dependencies have not finished'
assert not (D/'public-instruction.md').exists(), 'context transition already started; skip inspection'
script='''import importlib.metadata,importlib.util,json,pathlib,sys
p=pathlib.Path('/opt/ipfs-supervisor/datasets/ipfs_datasets_py/logic/software_contracts/content.py')
s=importlib.util.spec_from_file_location('isolated_cid_inspection',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
import multiformats
layout=m._native_registry_layout();m._memo_registration_key();key=m._memo_registration_key()
print(json.dumps(dict(python=sys.version.split()[0],module_version=multiformats.__version__,distribution_versions={n:importlib.metadata.version(n) for n in ('multiformats','multiformats-config','typing-validation','bases','duckdb')},native_layout_recognized=layout is not None,native_registration_matches_public_lookup=key==m._memo_registration_key_slow())))
'''
cmd=['docker','exec','ipfs-native-deploy-1791015700-main-1','/opt/ipfs-supervisor/venv/bin/python','-c',script]
(B/'docker-cid-command.json').write_text(json.dumps(dict(argv=cmd,scope='Separate read-only process before context transition; no runtime changes or measured-process memo priming.'),indent=2)+'\n')
start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
(B/'docker-cid.stdout').write_text(p.stdout);(B/'docker-cid.stderr').write_text(p.stderr)
r=dict(returncode=p.returncode,seconds=time.monotonic()-start,context_marker_present_after=(D/'public-instruction.md').exists())
if p.returncode==0:r['observation']=json.loads(p.stdout)
(B/'docker-cid-observation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
