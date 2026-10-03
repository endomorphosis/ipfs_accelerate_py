from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
A=WORKSPACE/'.worktrees/ir-release-accelerate-20261002'
D=WORKSPACE/'.worktrees/ir-release-datasets-20261002'
generation=sys.argv[1]
assert generation in ('before','proposed')
name='controls-'+generation
args=[sys.executable,'-B','-m','pytest','-c','/dev/null',str(ROOT/'tests'),'-q','--disable-warnings','--junitxml='+str(ROOT/(name+'.xml'))]
overrides={'PYTHONPATH':os.pathsep.join(map(str,[ROOT/'bootstrap',A,D])),
 'SOURCE384_LIFETIME_GENERATION':generation,'PYTHONDONTWRITEBYTECODE':'1',
 'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH':str(ROOT/(name+'-seal.db'))}
env=dict(os.environ,**overrides)
receipt={'argv':args,'cwd':str(ROOT),'environment_overrides':overrides,
 'scope':'Two complete canonical modules selected from off-tree files. Controlled loader/admission seams; logical object lifetime, not native inference or RSS.',
 'source_pins':json.loads((ROOT/'source-pins.json').read_text()),
 'test_pins':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'tests').glob('*.py'))}}
(ROOT/(name+'-command.json')).write_text(json.dumps(receipt,indent=2)+'\n')
start=time.monotonic()
with (ROOT/(name+'.log')).open('w') as log:result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt={'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start}
(ROOT/(name+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));print((ROOT/(name+'.log')).read_text()[-10000:])
raise SystemExit(result.returncode)
