"""Host-side exact compact edit replay on permitted public benchmark inputs."""
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys, time
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope, create_isolated_worktree
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews, canonical_json
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_edit_candidate import apply_source_view_edit_candidate

R=Path(__file__).resolve().parent
out=R/sys.argv[1];out.mkdir(exist_ok=False)
root=out/'canonical';root.mkdir()
public=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
sources={name:(public/name).read_bytes() for name in ('eigen.py','eval.py')}
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL)
for name,raw in sources.items():(root/name).write_bytes(raw)
git('init','-q');git('add','--all');git('-c','user.name=Qualification','-c','user.email=qualification@example.invalid','commit','-qm','Permitted public inputs')
store=WorktreeLifecycleStore(repo_root=root,store_dir=out/'lifecycle',lease_seconds=300)
t=time.monotonic()
with create_isolated_worktree(repo_root=root,worktree_path=out/'allocated',task_id='authored-compact-edit-probe',lifecycle_store=store) as owner:
 views=WorkerSourceViews(repository=root,workspace=owner.worktree_path,source_hashes={n:sha(b) for n,b in sources.items()},task_id=owner.task_id,scope_cid='scope:public-eigenval-fixture',semantic_root_cid='semantic:authored-fixture',semantic_sha256='a'*64,native_prompt_sha256='b'*64,source_manifest_sha256='c'*64,profile='python-comments@1')
 first=views.working_page(path='eigen.py')
 view=build_source_view(sources['eigen.py'],profile='python-comments@1').view
 before=b'idx = np.argmax(np.abs(eigenvalues))\n    return eigenvalues[idx], eigenvectors[:, idx]'
 after=before.replace(b'idx',b'i')
 start=view.index(before)
 receipt=apply_source_view_edit_candidate(owner=owner,views=views,scope=PatchScope(allowed_paths=('eigen.py',),task_owned_paths=('eigen.py',),effect_paths=('eigen.py',)),lease_id=owner.lease_id,fence=owner.fence,path='eigen.py',expected_workspace_generation=first['workspace_generation'],expected_source_sha256=first['source_sha256'],expected_view_sha256=first['view_sha256'],expected_manifest_sha256=first['manifest_sha256'],view_start=start,view_end=start+len(before),expected_view_preimage=before,replacement=after)
 candidate=(owner.worktree_path/'eigen.py').read_bytes()
 assert candidate==sources['eigen.py'].replace(before,after)
 assert (owner.worktree_path/'eval.py').read_bytes()==sources['eval.py']
 assert all((root/n).read_bytes()==b for n,b in sources.items())
 assert b'# beat this reference solution!' in candidate and b'# beat this reference solution!' not in view
 try:views.working_sources(expected_workspace_generation=first['workspace_generation'])
 except ValueError:stale_refused=True
 else:raise AssertionError('old generation accepted')
 # Authored deterministic checks on copied, reviewed public code. This is not
 # the official benchmark verifier or a claim of formal semantic equivalence.
 import numpy as np
 def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.find_dominant_eigenvalue_and_eigenvector
 old=load(root/'eigen.py','original_public_eigen');new=load(owner.worktree_path/'eigen.py','candidate_public_eigen')
 rng=np.random.default_rng(17)
 matrices=[np.diag([1.,-3.]),np.array([[0.,-2.],[2.,0.]]),*(rng.normal(size=(n,n)) for n in range(2,11))]
 for matrix in matrices:
  a,v=old(matrix);b,w=new(matrix)
  assert np.allclose(a,b) and np.allclose(v,w) and np.allclose(matrix@w,b*w)
 result=dict(schema='compact-source-edit-public-fixture-qualification@1',task='largest-eigenval',scope='Host owner application of a prescribed local variable rename, not a task solution or autonomous benchmark.',public_source_sha256={n:sha(b) for n,b in sources.items()},original_source_bytes=len(sources['eigen.py']),compact_view_bytes=len(view),candidate_source_bytes=len(candidate),edit_preimage_bytes=len(before),edit_replacement_bytes=len(after),hidden_comment_preserved=True,unrelated_file_unchanged=True,canonical_repository_unchanged=True,old_generation_refused=stale_refused,authored_numerical_cases_passed=len(matrices),official_verifier_executed=False,official_reward=None,full_supervisor_start=False,worker_socket_mutation=False,provider_calls=0,provider_tokens=None,token_savings_measured=False,source_obligations_discharged=0,application_receipt=receipt)
result.update(seconds=time.monotonic()-t,allocated_worktree_removed=not (out/'allocated').exists())
(out/'qualification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='application_receipt'},sort_keys=True))
