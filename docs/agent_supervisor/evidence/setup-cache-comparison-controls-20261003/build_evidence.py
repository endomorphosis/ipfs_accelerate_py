from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent;A=B.parents[1]/'.worktrees/ir-release-accelerate-20261002'
E=A/'docs/agent_supervisor/evidence/setup-cache-comparison-controls-20261003'
assert not E.exists();E.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,r):
 out=E/r;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
for p in (B/'final-sources').glob('*.py'):copy(p,'final-sources/'+p.name)
copy(B/'before/benchmark_controls.py','before/benchmark_controls.py')
for name in ('legacy-golden-before.json','correction.json','prepare.stderr','prepare-exit.json','controls-01.log','controls-01-command.json','controls-01-exit.json','controls-02.log','controls-02.xml','controls-02-command.json','controls-02-exit.json','run_controls-01.py','run_controls.py','build_evidence.py'):
 copy(B/name,name)
x=ET.parse(B/'controls-02.xml');assert len(x.findall('.//testcase'))==59 and not any(x.findall('.//'+k) for k in ('failure','error','skipped'))
(E/'README.md').write_text('''# Setup-cache comparison-control declaration

The first full-trial preparation stopped on the host because the closed
supervisor-kwargs allowlist did not yet recognize `setup_cache_selection`.
The exact traceback and prepare exit are retained. It failed before task/agent
execution; no provider or model call was launched by that preparation.

The additive correction accepts only the existing closed selection schema/policy
and a 64-character lowercase manifest digest, on the full supervisor arm with
the supported common resource profile. Existing resource-limit validation still
runs. Explicit `None`, malformed or extra selection fields, no-index and missing
or incompatible profiles refuse. This pure declaration code performs no archive
or credential reads; deployment owners retain their separate validation.

The declaration schema and serialized shape are unchanged. The existing complete
configuration digest binds every adapter kwarg, including the selected policy and
manifest. Changing or dropping a selection after preparation yields mismatch.
Absent-selection legacy declarations retain their exact digests. Comparison still
means equality of the previously declared common configuration fields only;
unequal adapter setup work and runtime enforcement are not established by it.

**59 actual host-side tests passed in 20.85 seconds**, with no skips. They include
existing comparison controls, actual Harbor config normalization, six fixed
pre-change digest cases and the new shape/profile/tamper controls. The first test
launcher found no pytest in the Harbor virtualenv and collected no tests. Its
failure is retained; the passing launcher supplied existing installed host pytest
paths without installing dependencies or changing source.

Only `benchmark_controls.py` and its new focused test changed. The controls owner
is imported by host benchmark preparation, baseline and comparison code. The host
Harbor adapter also imports their constants/helpers. The container driver,
qualification probe and preparation/deployment owners do not import those host
controllers; all 33 selected runtime source pins are unchanged. Reusing the
qualified container archive therefore does not silently replace runtime owners.

This package contains exact before/final source, raw controls, pre-change golden
declarations, the retained preparation error and source-free scope metadata. It
contains no auth contents, task/verifier source, model weights, database or
runtime archive. No new benchmark reward or token score is claimed here.
''')
m={'schema':'setup-cache-comparison-controls-evidence@1','files':[{'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(E.rglob('*')) if p.is_file()]}
(E/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
source=['benchmarks/agent_supervisor/container_coding/benchmark_controls.py','benchmarks/agent_supervisor/container_coding/test_setup_cache_comparison_controls.py']
scope=sorted(source+[str(p.relative_to(A)) for p in E.rglob('*') if p.is_file()])
(B/'commit-scope.json').write_text(json.dumps({'repository':str(A),'files':[{'path':s,'bytes':(A/s).stat().st_size,'sha256':sha(A/s)} for s in scope]},indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(E/'manifest.json'),'members':len(m['files']),'member_bytes':sum(r['bytes'] for r in m['files']),'scope_files':len(scope)},indent=2))
