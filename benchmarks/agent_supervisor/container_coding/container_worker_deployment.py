"""Install and qualify the separate model-worker identity in a Harbor container.

All scripts and the namespace manifest are owned by container root. Model calls
are absent from qualification; the only delegated entry is the pinned router.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from .terminal_deployment import PYTHON, ROOT, native_codex_exposure_script, runtime_environment

def prompt_tokenizer_arguments(*, prompt_tokenizer_artifact: str | Path | None = None,
                              prompt_tokenizer_sha256: str | None = None,
                              max_input_tokens: int | None = None) -> list[str]:
    """Forward explicit child-local inputs; the router still admits their bytes.

    Without a descriptor the child chooses its automatic offline counter. This
    boundary neither reads nor stages assets. An absolute descriptor path
    is required because planning and coding use separate allocated worktrees.
    The existing child loader checks the exact provider/model, descriptor,
    tokenizer asset, runtime, and complete untruncated prompt.
    """
    supplied = prompt_tokenizer_artifact is not None or prompt_tokenizer_sha256 is not None
    if supplied and (prompt_tokenizer_artifact is None or prompt_tokenizer_sha256 is None):
        raise ValueError("exact prompt tokenizer descriptor and digest required")
    if max_input_tokens is not None:
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("positive complete prompt token budget required")
    if not supplied:
        return ["--max-input-tokens", str(max_input_tokens)] if max_input_tokens is not None else []
    if not isinstance(prompt_tokenizer_artifact, str | Path):
        raise ValueError("explicit absolute child-local tokenizer path required")
    text = str(prompt_tokenizer_artifact)
    path = Path(text)
    if (not text or len(text.encode()) > 4096 or not path.is_absolute()
            or text != str(path) or "\x00" in text or ":" in text or "\\" in text
            or any(part in {".", ".."} for part in text.split("/"))):
        raise ValueError("explicit absolute child-local tokenizer path required")
    if (type(prompt_tokenizer_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", prompt_tokenizer_sha256) is None):
        raise ValueError("exact prompt tokenizer descriptor and digest required")
    return ["--prompt-tokenizer-artifact", text,
            "--prompt-tokenizer-sha256", prompt_tokenizer_sha256,
            *(["--max-input-tokens", str(max_input_tokens)] if max_input_tokens is not None else [])]

def semantic_context_arguments(*, semantic_slot_policy_revision: str = "legacy@1",
                               semantic_context_projection: bool = False,
                               semantic_native_input: bool = False) -> list[str]:
    """Encode explicit comparison options without selecting a model input.

    Callers must additionally require their bound semantic model coding route.
    The child router retains source admission, complete token comparison, and
    native fallback. The native control preserves the complete semantic input
    and excludes projection and identifier encoding. No assets are supplied here.
    """
    if type(semantic_slot_policy_revision) is not str or semantic_slot_policy_revision not in {
        "legacy@1", "closed-typed-identifiers@2",
    }:
        raise ValueError("unsupported semantic identifier slot policy")
    if type(semantic_context_projection) is not bool:
        raise ValueError("semantic context projection must be an explicit Boolean")
    if type(semantic_native_input) is not bool:
        raise ValueError("semantic native input must be an explicit Boolean")
    if semantic_native_input and (semantic_context_projection or semantic_slot_policy_revision != "legacy@1"):
        raise ValueError("semantic native input requires exclusive legacy semantic context")
    if semantic_context_projection and semantic_slot_policy_revision != "closed-typed-identifiers@2":
        raise ValueError("semantic context projection requires the versioned identifier policy")
    return [*(["--semantic-slot-policy-revision", semantic_slot_policy_revision]
              if semantic_slot_policy_revision != "legacy@1" else []),
            *(["--semantic-context-projection"] if semantic_context_projection else []),
            *(["--semantic-native-input"] if semantic_native_input else [])]

def semantic_decision_arguments(*,
        semantic_decision_policy_artifact: str | Path | None = None,
        semantic_decision_policy_sha256: str | None = None,
        semantic_decision_nomination_artifact: str | Path | None = None,
        semantic_decision_nomination_sha256: str | None = None,
        disable_semantic_typed_nomination: bool = False,
        semantic_retrieval_query_reference: bool = False,
        owner_binding: bool = False,
        semantic_decision_max_calls: int = 3,
        semantic_decision_max_total_input_tokens: int | None = None) -> list[str]:
    """Encode a prebound policy or independently pinned owner nomination.

    The worker admits prebound descriptors against native task context. A
    nomination is materialized by the owner against the final compiled stdin;
    its admitted task key must be added by the signing caller. This function
    neither loads nor stages assets. Model output never selects policy.
    """
    if (type(disable_semantic_typed_nomination) is not bool or type(owner_binding) is not bool
            or type(semantic_retrieval_query_reference) is not bool):
        raise ValueError("typed nomination switches must be boolean")
    if type(semantic_decision_max_calls) is not int or not 1 <= semantic_decision_max_calls <= 4:
        raise ValueError("semantic decision call limit must be between 1 and 4")
    if (semantic_decision_max_total_input_tokens is not None
            and (type(semantic_decision_max_total_input_tokens) is not int
                 or semantic_decision_max_total_input_tokens < 1)):
        raise ValueError("positive semantic decision cumulative input budget required")
    policy_supplied = semantic_decision_policy_artifact is not None or semantic_decision_policy_sha256 is not None
    nomination_supplied = (semantic_decision_nomination_artifact is not None
                           or semantic_decision_nomination_sha256 is not None)
    if semantic_retrieval_query_reference and not nomination_supplied:
        raise ValueError("retrieval query reference requires an independently pinned owner nomination")
    if (disable_semantic_typed_nomination or owner_binding) and not nomination_supplied:
        raise ValueError("typed nomination switches require an owner nomination")
    if policy_supplied and nomination_supplied:
        raise ValueError("choose either a prebound semantic decision policy or an owner nomination")
    supplied = policy_supplied or nomination_supplied
    if not supplied:
        if semantic_decision_max_calls != 3 or semantic_decision_max_total_input_tokens is not None:
            raise ValueError("semantic decision budgets require an explicit policy")
        return []
    # Reuse the closed child-local spelling/hash checks; this does not load a
    # tokenizer or assert compatibility of a policy with any provider.
    try:
        pair = prompt_tokenizer_arguments(
            prompt_tokenizer_artifact=(semantic_decision_nomination_artifact if nomination_supplied
                                       else semantic_decision_policy_artifact),
            prompt_tokenizer_sha256=(semantic_decision_nomination_sha256 if nomination_supplied
                                     else semantic_decision_policy_sha256))
    except ValueError as exc:
        raise ValueError("exact absolute semantic decision policy and digest required") from exc
    kind = "nomination" if nomination_supplied else "policy"
    return ["--semantic-decision-" + kind + "-artifact", pair[1],
            "--semantic-decision-" + kind + "-sha256", pair[3],
            *(["--semantic-decision-typed-nomination",
                "disabled" if disable_semantic_typed_nomination else "enabled"]
              if owner_binding else
              ["--disable-semantic-typed-nomination"] if disable_semantic_typed_nomination else []),
            "--semantic-decision-max-calls", str(semantic_decision_max_calls),
            *(["--semantic-decision-max-total-input-tokens", str(semantic_decision_max_total_input_tokens)]
              if semantic_decision_max_total_input_tokens is not None else [])]

def doctor_dispatch_arguments(*, arm: str, doctor_dispatch_policy: str = "auto") -> list[str]:
    """Forward an explicit full-arm model-route ablation without authority."""
    if type(arm) is not str or arm not in {"full", "no-index"}:
        raise ValueError("explicit supervisor arm required")
    if type(doctor_dispatch_policy) is not str or doctor_dispatch_policy not in {"auto", "model-only"}:
        raise ValueError("unsupported Doctor dispatch policy")
    if doctor_dispatch_policy == "model-only" and arm != "full":
        raise ValueError("model-only Doctor dispatch policy requires the full indexed arm")
    return ["--doctor-dispatch-policy", doctor_dispatch_policy] if doctor_dispatch_policy != "auto" else []


def validate_legacy_semantic_route(*, policyoptions: dict, rendereroptions: dict) -> None:
    """Refuse unqualified rendering combinations for selected semantic routes."""
    if (type(policyoptions) is not dict or type(rendereroptions) is not dict
            or len(policyoptions) > 32 or len(rendereroptions) > 32
            or any(type(key) is not str for key in (*policyoptions, *rendereroptions))):
        raise ValueError("bounded literal semantic route option dictionaries required")
    decision_selected = (policyoptions.get("semantic_retrieval_query_reference", False) is not False
        or any(policyoptions.get(key) is not None for key in (
            "semantic_decision_policy_artifact", "semantic_decision_policy_sha256",
            "semantic_decision_nomination_artifact", "semantic_decision_nomination_source",
            "semantic_decision_nomination_sha256")))
    selected = (decision_selected
        or policyoptions.get("semantic_slot_policy_revision", "legacy@1") != "legacy@1"
        or policyoptions.get("semantic_context_projection", False) is not False
        or policyoptions.get("semantic_native_input", False) is not False)
    if not selected:
        return
    defaults = {
        "semantic_transport_schema": "supervisor-semantic-router-input@1",
        "semantic_metadata_view": "legacy", "semantic_working_set": "legacy",
        "source_view_profile": "disabled", "source_edit_mode": "disabled",
        "source_view_policy": None, "coding_reply_mode": "legacy",
        "tool_output_policy": "legacy", "source_view_advisory_policy": "legacy",
        "repository_source_policy": "legacy", "source_input_policy": "legacy",
        "query_reference_policy": "legacy", "semantic_context_profile": "legacy-bounded@1",
        "capsule_selection": "lexical-query-symbol-name@1", "capsule_budget": "legacy",
        "metadata_census_only": False,
    }
    if any(key not in defaults and key != "provider" for key in rendereroptions):
        raise ValueError("unknown semantic renderer option")
    provider = rendereroptions.get("provider", "codex_cli")
    if decision_selected and (type(provider) is not str or provider != "codex_cli"):
        raise ValueError("semantic decision policy requires the qualified Codex provider")
    incompatible = sorted(key for key, default in defaults.items()
        if key in rendereroptions and (type(rendereroptions[key]) is not type(default)
                                      or rendereroptions[key] != default))
    if incompatible:
        raise ValueError("native/decision semantic routes require separately qualified renderer controls: "
                         + ", ".join(incompatible))


SOURCE_LOOKUP_ENTRY = r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import os,pathlib,runpy,stat,sys
root=pathlib.Path('/opt/ipfs-supervisor')
try:
 paths=[pathlib.Path(__file__).absolute(),root/'source',root/'datasets']
 for path in paths:
  if path.resolve(strict=True)!=path:raise ValueError('exact protected import required')
  for parent in (path,*path.parents):
   info=parent.lstat()
   if info.st_uid!=0 or info.st_mode&0o022:raise ValueError('deployer protection required')
 if not all(path.is_dir() for path in paths[1:]):raise ValueError('deployed import roots required')
 sys.path[:0]=[str(root/'source'),str(root/'datasets')]
 runpy.run_module('ipfs_accelerate_py.agent_supervisor.runtime.source_lookup_worker_client',run_name='__main__')
except (OSError,ValueError,ImportError,RuntimeError):
 print('{"reason_code":"transport_error"}',file=sys.stderr)
 raise SystemExit(2)
"""

OWNER_MONITOR = r"""
import json,signal,subprocess,time
process=None
worker_waited=False
worker_cleanup_succeeded=False
def interrupted(signum,frame):
 raise InterruptedError('owner monitor interrupted')
for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
code=1
try:
 process=subprocess.Popen(command)
 broker=globals().get("numerical_obligation_broker")
 if broker is not None:broker.note_spawned(process)
 source_broker=globals().get("worker_source_broker")
 if source_broker is not None:source_broker.note_spawned(process)
 repository_owner=globals().get("repository_source_owner")
 if repository_owner is not None:repository_owner.note_spawned(process)
 guard=globals().get("proof_query_guard")
 if guard is not None:guard.note_spawned(process)
 code=process.wait()
 worker_waited=True
except BaseException:
 if process is not None:
  try:process.terminate()
  except ProcessLookupError:pass
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:process.kill();process.wait()
 raise
finally:
 for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,signal.SIG_IGN)
 try:
  cleanup=subprocess.run(['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry','--cleanup'],cwd='/',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,start_new_session=True,timeout=15)
  worker_cleanup_succeeded=cleanup.returncode==0
  if cleanup.returncode:code=1
 finally:
  try:
   guard=globals().get("proof_query_guard")
   if guard is not None:guard.abort()
  finally:
   try:
    broker=globals().get("numerical_obligation_broker")
    if broker is not None:broker.close()
   finally:
    try:
     source_broker=globals().get("worker_source_broker")
     if source_broker is not None:source_broker.close()
    finally:
     repository_owner=globals().get("repository_source_owner")
     if repository_owner is not None:
      from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_monitor import retire_repository_source_owner
      repository_source_retirement=retire_repository_source_owner(repository_owner)
# Retired worker observations cannot authorize another child or completion.
# Refresh only after native wait, isolated cleanup and every broker retirement.
source_broker=globals().get("worker_source_broker")
refresh_deadline=globals().get("owner_candidate_round_deadline")
if worker_waited and worker_cleanup_succeeded and type(code) is int and code>0 and source_broker is not None and refresh_deadline is not None:
 observation=None
 refresh_flags={'proof_authority':False,'execution_authority':False,'completion_authority':False,'retry_authority':False,'settlement_authority':False,'automatic_retry_admitted':False,'model_inference_calls':0,'provider_calls':0,'source_obligations_discharged':0}
 try:
  # Cleanup must finish, but advisory candidate analysis remains interruptible.
  for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
  if type(code) is not int or process.returncode!=code or process.poll()!=code:
   raise ValueError('worker exit is not settled')
  if time.monotonic()>=refresh_deadline:
   observation={'schema':'owner-candidate-refresh-observation@1','status':'skipped','reason_code':'deadline_exhausted',**refresh_flags}
  else:
   from ipfs_accelerate_py.agent_supervisor.runtime.owner_candidate_round_checkpoint import persist_failed_worker_candidate_context
   observation=persist_failed_worker_candidate_context(source_broker=source_broker,prior_returncode=code,absolute_deadline=refresh_deadline)
 except Exception:
  observation={'schema':'owner-candidate-refresh-observation@1','status':'failed','reason_code':'refresh_failed',**refresh_flags}
 try:
  if observation is not None:print('[owner-candidate-refresh] '+json.dumps(observation,sort_keys=True,allow_nan=False),flush=True)
 except Exception:
  pass
raise SystemExit(code)
"""

# Both deployed identities must refuse this route until an owner-backed source
# transport is qualified. The direct trusted Python router API remains separate.
# Inspect every occurrence, including argparse long-option abbreviations: a
# later legacy selection must not hide an earlier unsupported selection.
_REPOSITORY_SOURCE_ISOLATION_GUARD = r"""
for source_policy_index,source_policy_argument in enumerate(sys.argv[1:],1):
 source_policy_flag,source_policy_separator,source_policy_value=source_policy_argument.partition('=')
 if source_policy_flag.startswith('--') and len(source_policy_flag)>2 and '--repository-source-policy'.startswith(source_policy_flag):
  if not source_policy_separator:source_policy_value=sys.argv[source_policy_index+1] if source_policy_index+1<len(sys.argv) else None
  if source_policy_value!='legacy':raise SystemExit('repository_source_owner_transport_unavailable')
"""

OWNER_ENTRY = (
    r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import json,os,pathlib,sys,time
owner_candidate_round_started=time.monotonic()
root=pathlib.Path('/opt/ipfs-supervisor')
boundary=json.loads((root/'container-boundary.json').read_text())
if os.getuid()!=boundary['owner_uid'] or os.geteuid()!=boundary['owner_uid']:raise SystemExit('owner identity required')
"""
    + _REPOSITORY_SOURCE_ISOLATION_GUARD
    + r"""cwd=pathlib.Path.cwd()
if cwd.resolve()!=cwd or not any(cwd!=pathlib.Path(p) and cwd.is_relative_to(p) for p in boundary['allowed_worktree_roots']):raise SystemExit('allocated candidate required')
marker=cwd/'.git'
if not marker.is_file() or marker.is_symlink() or marker.stat().st_uid!=1000:raise SystemExit('owner allocated Git worktree marker required')
# Allocate group writes only in this exact candidate; canonical and owner state
# remain outside this traversal. Refuse links rather than chmod their targets.
for base,dirs,files in os.walk(cwd,followlinks=False):
 p=pathlib.Path(base)
 if p.is_symlink() or p.stat().st_uid not in (1000,1001):raise SystemExit('candidate directory ownership differs')
 if p.stat().st_uid==1000:p.chmod(0o3770 if p==cwd else 0o2770)
 for name in dirs:
  if (p/name).is_symlink():raise SystemExit('candidate directory symlink refused')
 for name in files:
  q=p/name
  if q.is_symlink() or not q.is_file() or q.stat().st_uid not in (1000,1001) or q.stat().st_nlink!=1:raise SystemExit('candidate input is not one allocated regular file')
  if q.stat().st_uid==1000:q.chmod(0o440 if q==cwd/'.git' else (0o770 if q.stat().st_mode&0o111 else 0o660))
sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
# The fixed lease includes allocation checks, ownership setup and formalization.
from ipfs_accelerate_py.agent_supervisor.runtime.numerical_obligation_broker import _argument,MAX_SESSION_SECONDS
owner_candidate_round_deadline=None
try:
 owner_candidate_round_timeout=int(_argument(list(sys.argv[1:]),'--timeout','90'))
 if 1<=owner_candidate_round_timeout<=MAX_SESSION_SECONDS:owner_candidate_round_deadline=owner_candidate_round_started+owner_candidate_round_timeout
except (TypeError,ValueError,OverflowError):
 pass
worker_arguments=sys.argv[1:]
owner_prompt_bytes=None
if any(part=='--semantic-native-input' or part.startswith('--semantic-native-input=') for part in worker_arguments) and any(part==flag or part.startswith(flag+'=') for part in worker_arguments for flag in ('--semantic-decision-policy-artifact','--semantic-decision-policy-sha256','--semantic-decision-nomination-artifact','--semantic-decision-nomination-sha256','--semantic-decision-task-id','--semantic-decision-typed-nomination','--semantic-decision-context-bundle-artifact','--semantic-decision-context-bundle-sha256','--semantic-decision-task-cid')):raise SystemExit('semantic native input excludes semantic decision policy and nomination')
# Capture original stdin once through the existing bounded sealed handoff.
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import prepare_container_source_formalization_dispatch
handoff=prepare_container_source_formalization_dispatch(argv=worker_arguments,boundary=root/'container-boundary.json',workspace=cwd)
os.dup2(handoff['stdin_fd'],0,inheritable=True);os.close(handoff['stdin_fd'])
worker_arguments=handoff['argv']
if any(part==flag or part.startswith(flag+'=') for part in worker_arguments for flag in ('--semantic-decision-nomination-artifact','--semantic-decision-nomination-sha256','--semantic-decision-task-id','--semantic-decision-typed-nomination','--semantic-decision-context-bundle-artifact','--semantic-decision-context-bundle-sha256','--semantic-decision-task-cid')):
 sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
 from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_owner import prepare_decision_owner_dispatch
 try:
  owner_prompt_bytes=os.pread(0,256001,0)
  dispatch=prepare_decision_owner_dispatch(arguments=worker_arguments,prompt_bytes=owner_prompt_bytes,workspace=cwd)
  if dispatch.prompt_bytes!=owner_prompt_bytes:raise ValueError('owner dispatch changed native stdin')
 except Exception as error:
  print(json.dumps({'schema':'supervisor-semantic-decision-owner-refusal@1','error_type':type(error).__name__,'provider_called':False,'completion_authority':False}),flush=True)
  raise SystemExit(1)
 worker_arguments=list(dispatch.worker_arguments)
 owner_prompt_bytes=dispatch.prompt_bytes
 print(json.dumps(dispatch.receipt,sort_keys=True,allow_nan=False),flush=True)
command=['/usr/bin/sudo','-n','-u','benchmarkworker','--',str(root/'bin/worker-entry'),*worker_arguments]
proof_query_guard=None
# Brokers receive the owner-bound argv; the worker inherits the same sealed fd0.
handoff={**handoff,'argv':worker_arguments}
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_broker import worker_source_argv
command=['/usr/bin/sudo','-n','-u','benchmarkworker','--',str(root/'bin/worker-entry'),*worker_source_argv(handoff['argv'])]
if any(part==flag or part.startswith(flag+'=') for part in sys.argv[1:] for flag in ('--finite-proof-query-context','--finite-proof-query-sha256','--finite-proof-query-context-cid')):
 sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_dispatch import require_finite_proof_query_isolated_worker_dispatch
 proof_query_guard=require_finite_proof_query_isolated_worker_dispatch(command=command,environment=dict(os.environ),worktree=cwd)
numerical_obligation_broker=None
worker_source_broker=None
repository_source_owner=None
try:
 from ipfs_accelerate_py.agent_supervisor.runtime.numerical_obligation_broker import prepare_numerical_obligation_broker
 numerical_obligation_broker=prepare_numerical_obligation_broker(argv=handoff['argv'],boundary=root/'container-boundary.json',workspace=cwd,input_fd=0)
 if numerical_obligation_broker is not None:command.extend(numerical_obligation_broker.extra_argv)
 from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_broker import prepare_worker_source_broker
 worker_source_broker=prepare_worker_source_broker(argv=handoff['argv'],boundary=root/'container-boundary.json',workspace=cwd,input_fd=0)
 if worker_source_broker is not None:command.extend(worker_source_broker.extra_argv)
 from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_dispatch import configure_repository_source_owner_dispatch
 repository_source_owner=configure_repository_source_owner_dispatch(argv=handoff['argv'],boundary=root/'container-boundary.json',workspace=cwd,deadline_monotonic=owner_candidate_round_deadline)
 if repository_source_owner is not None:
  import signal
  def repository_source_interrupted(signum,frame):
   raise InterruptedError('repository source owner interrupted')
  for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,repository_source_interrupted)
  from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_monitor import retire_repository_source_owner
  repository_source_owner.start()
  command.extend(repository_source_owner.extra_argv)
except BaseException:
 if repository_source_owner is not None:
  for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,signal.SIG_IGN)
 try:
  if proof_query_guard is not None:proof_query_guard.abort()
 finally:
  try:
   if numerical_obligation_broker is not None:numerical_obligation_broker.close()
  finally:
   try:
    if worker_source_broker is not None:worker_source_broker.close()
   finally:
    if repository_source_owner is not None:
     from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_monitor import retire_repository_source_owner
     repository_source_retirement=retire_repository_source_owner(repository_source_owner)
 raise
"""
    + OWNER_MONITOR
)

VALIDATION_ENTRY = (
    r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import os,sys
if os.getuid()!=1000 or os.geteuid()!=1000:raise SystemExit('owner identity required')
if len(sys.argv)<2:raise SystemExit('literal validation argv required')
command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry','--validate',*sys.argv[1:]]
"""
    + OWNER_MONITOR
)

WORKER_ENTRY = (r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import argparse,ctypes,hashlib,json,os,pathlib,sys,time
root=pathlib.Path('/opt/ipfs-supervisor')
if os.getuid()!=1001 or os.geteuid()!=1001:raise SystemExit('worker identity required')
# Prevent this external-isolation worker from acquiring any new Unix privilege.
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no-new-privileges failed')
os.umask(0o007)
os.environ.clear()
os.environ.update({'HOME':str(root/'worker-home'),'PATH':str(root/'provider-bin')+':/usr/local/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root/'source')+':'+str(root/'datasets')+':'+str(root/'kit'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
os.environ.update({'PYTHON':str(root/'venv/bin/python'),'PYTHON_BIN':str(root/'venv/bin/python'),'GIT_CONFIG_COUNT':'3','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':str(pathlib.Path.cwd()),'GIT_CONFIG_KEY_1':'core.hooksPath','GIT_CONFIG_VALUE_1':'/dev/null','GIT_CONFIG_KEY_2':'core.fsmonitor','GIT_CONFIG_VALUE_2':'false'})
os.environ.update({'IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE':str(root/'venv/bin/python'),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONHASHSEED':'0'})
sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
from ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary import verify_container_worker_boundary
artifact=root/'container-boundary.json'
digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
selected_provider=json.loads(artifact.read_bytes()).get('provider','codex_cli')
if selected_provider not in ('codex_cli','grok_cli'):raise SystemExit('unknown deployed provider')
if selected_provider=='grok_cli':
 os.environ['GROK_HOME']=str(root/'worker-home/.grok')
 os.environ['ipfs_accelerate_py_GROK_CLI_CMD']=str(root/'provider-bin/grok')
 os.environ.update({name:'0' for name in ('GROK_CODEX_AGENTS_ENABLED','GROK_CODEX_HOOKS_ENABLED','GROK_CODEX_MCPS_ENABLED','GROK_CODEX_RULES_ENABLED','GROK_CODEX_SESSIONS_ENABLED','GROK_CODEX_SKILLS_ENABLED')})
if sys.argv[1:]==['--cleanup']:
 boundary=json.loads(artifact.read_bytes())
 if boundary['namespaces']!={k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')}:raise SystemExit('cleanup namespace differs')
 import signal
 def workers():
  found=[]
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit() or int(p.name)==os.getpid():continue
   try:
    status=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
    if status['Uid'].split()[0]=='1001' and not status['State'].strip().startswith('Z'):found.append(int(p.name))
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  return found
 for signum in (signal.SIGTERM,signal.SIGKILL):
  for pid in workers():
   try:os.kill(pid,signum)
   except ProcessLookupError:pass
  time.sleep(.2)
 if workers():raise SystemExit('worker cleanup incomplete')
 raise SystemExit(0)
validating=len(sys.argv)>1 and sys.argv[1]=='--validate'
evidence=verify_container_worker_boundary(artifact=artifact,expected_sha256=digest,workspace=pathlib.Path.cwd(),purpose='validation' if validating else 'coding')
if validating:
 if len(sys.argv)<3 or len(sys.argv)>130 or any(len(x)>8192 for x in sys.argv[2:]):raise SystemExit('bounded literal validation argv required')
 os.execvpe(sys.argv[2],sys.argv[2:],dict(os.environ))
parser=argparse.ArgumentParser()
parser.add_argument('--provider',choices=['codex_cli','grok_cli'],default=selected_provider)
parser.add_argument('--model',default='gpt-6.1-sol');parser.add_argument('--reasoning-effort',choices=['low','medium','high','xhigh','max'],default='high')
parser.add_argument('--timeout',type=int,default=90);parser.add_argument('--max-output-tokens',type=int,default=4096)
parser.add_argument('--budget-profile')
parser.add_argument('--purpose',choices=['planning','coding'],default='coding')
parser.add_argument('--semantic-repository',type=pathlib.Path)
parser.add_argument('--semantic-slot-policy-revision',choices=['legacy@1','closed-typed-identifiers@2'],default='legacy@1')
parser.add_argument('--semantic-context-projection',action='store_true')
parser.add_argument('--semantic-native-input',action='store_true')
parser.add_argument('--semantic-decision-policy-artifact',type=pathlib.Path)
parser.add_argument('--semantic-decision-policy-sha256')
parser.add_argument('--semantic-decision-max-calls',type=int,default=3)
parser.add_argument('--semantic-decision-max-total-input-tokens',type=int)
parser.add_argument('--prompt-tokenizer-artifact',type=pathlib.Path)
parser.add_argument('--prompt-tokenizer-sha256')
parser.add_argument('--max-input-tokens',type=int)
parser.add_argument('--semantic-transport-schema',choices=['supervisor-semantic-router-input@1','supervisor-semantic-router-input@2'],default='supervisor-semantic-router-input@1')
parser.add_argument('--semantic-metadata-view',choices=['legacy','common-bindings@1','common-bindings-instruction@1'],default='legacy')
parser.add_argument('--semantic-working-set',choices=['legacy','capsule-references@1'],default='legacy')
parser.add_argument('--semantic-metadata-census-only',action='store_true')
parser.add_argument('--coding-reply-mode',choices=['legacy','ordinary-completion@1'],default='legacy')
parser.add_argument('--tool-output-policy',choices=['legacy','artifact-preview@1','artifact-preview@2'],default='legacy')
parser.add_argument('--source-view-advisory-policy',choices=['legacy','source-view-raw@1','source-view-compact@1'],default='legacy')
# Preserve the policy's previously unique argparse abbreviations after adding
# the public owner pair; --re was already ambiguous with --reasoning-effort.
repository_source_policy_options=tuple('--repository-source-policy'[:size] for size in range(len('--rep'),len('--repository-source-policy')+1))
parser.add_argument(*repository_source_policy_options,dest='repository_source_policy',choices=['legacy','repository-source-raw@1','repository-source-compact@1'],default='legacy')
parser.add_argument('--repository-source-owner-descriptor')
parser.add_argument('--repository-source-owner-descriptor-sha256')
parser.add_argument('--source-input-policy',choices=['legacy','source-input-raw@1','source-input-compact@1','source-input-ast-ref-raw@1','source-input-ast-ref-compact@1'],default='legacy')
parser.add_argument('--query-reference-policy',choices=['legacy','required-instruction-ref@1'],default='legacy')
parser.add_argument('--doctor-candidate-artifact',type=pathlib.Path)
parser.add_argument('--doctor-candidate-sha256')
parser.add_argument('--doctor-task-cid')
parser.add_argument('--doctor-contract-artifact',type=pathlib.Path)
parser.add_argument('--doctor-contract-sha256')
parser.add_argument('--doctor-contract-task-cid')
parser.add_argument('--finite-repository-artifact',type=pathlib.Path)
parser.add_argument('--finite-repository-sha256')
parser.add_argument('--finite-repository-task-cid')
parser.add_argument('--finite-proof-query-context',type=pathlib.Path)
parser.add_argument('--finite-proof-query-sha256')
parser.add_argument('--finite-proof-query-context-cid')
parser.add_argument('--doctor-residual-artifact',type=pathlib.Path)
parser.add_argument('--doctor-residual-sha256')
parser.add_argument('--doctor-residual-task-cid')
parser.add_argument('--public-instruction-artifact',type=pathlib.Path)
parser.add_argument('--public-instruction-sha256')
parser.add_argument('--public-instruction-task-cid')
parser.add_argument('--source-formalization-handoff',type=pathlib.Path)
parser.add_argument('--source-formalization-handoff-sha256')
parser.add_argument('--numerical-obligation-descriptor',type=pathlib.Path)
parser.add_argument('--numerical-obligation-descriptor-sha256')
parser.add_argument('--source-lookup-descriptor',type=pathlib.Path)
parser.add_argument('--source-lookup-descriptor-sha256')
parser.add_argument('--grok-coding-max-turns',type=int,default=128)
parser.add_argument('--source-view-profile',default='disabled',choices=('disabled','identity@1','python-whitespace@1','python-comments@1'))
parser.add_argument('--source-edit-mode',default='disabled',choices=('disabled','mapped-span@1','mapped-span@2'))
parser.add_argument('--source-view-descriptor',type=pathlib.Path)
parser.add_argument('--source-view-descriptor-sha256')
parser.add_argument('--preflight',action='store_true');parser.add_argument('--preflight-sleep',type=int,default=0)
# Inspect original spellings before argparse can normalize or overwrite them.
# Only the owner creates this public pair; private native grants have no CLI.
repository_source_owner_options=('--repository-source-owner-descriptor','--repository-source-owner-descriptor-sha256')
repository_source_owner_seen=set()
for repository_source_owner_argument in sys.argv[1:]:
 repository_source_owner_flag=repository_source_owner_argument.partition('=')[0]
 if repository_source_owner_flag in repository_source_owner_options:
  if repository_source_owner_flag in repository_source_owner_seen:raise SystemExit('duplicate repository source owner option')
  repository_source_owner_seen.add(repository_source_owner_flag)
 elif repository_source_owner_flag.startswith('--') and len(repository_source_owner_flag)>2 and repository_source_owner_flag not in repository_source_policy_options and any(option.startswith(repository_source_owner_flag) for option in repository_source_owner_options):raise SystemExit('exact repository source owner option required')
args=parser.parse_args()
if args.provider!=selected_provider:raise SystemExit('provider differs from deployed worker route')
doctor_values=(args.doctor_candidate_artifact,args.doctor_candidate_sha256,args.doctor_task_cid)
residual_values=(args.doctor_residual_artifact,args.doctor_residual_sha256,args.doctor_residual_task_cid)
contract_values=(args.doctor_contract_artifact,args.doctor_contract_sha256,args.doctor_contract_task_cid)
finite_values=(args.finite_repository_artifact,args.finite_repository_sha256,args.finite_repository_task_cid)
proof_query_values=(args.finite_proof_query_context,args.finite_proof_query_sha256,args.finite_proof_query_context_cid)
instruction_values=(args.public_instruction_artifact,args.public_instruction_sha256,args.public_instruction_task_cid)
formalization_values=(args.source_formalization_handoff,args.source_formalization_handoff_sha256)
numerical_obligation_values=(args.numerical_obligation_descriptor,args.numerical_obligation_descriptor_sha256)
source_lookup_values=(args.source_lookup_descriptor,args.source_lookup_descriptor_sha256)
if any(value is not None for value in source_lookup_values) and (not all(source_lookup_values) or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or any(contract_values) or any(doctor_values) or any(finite_values) or any(proof_query_values)):raise SystemExit('exact source lookup descriptor requires isolated router coding')
source_view_values=(args.source_view_descriptor,args.source_view_descriptor_sha256)
tokenizer_values=(args.prompt_tokenizer_artifact,args.prompt_tokenizer_sha256)
decision_values=(args.semantic_decision_policy_artifact,args.semantic_decision_policy_sha256)
decision_options=any(value is not None for value in decision_values)
semantic_options=args.semantic_native_input or args.semantic_context_projection or args.semantic_slot_policy_revision!='legacy@1'
if args.semantic_native_input and (args.semantic_context_projection or args.semantic_slot_policy_revision!='legacy@1'):raise SystemExit('semantic native input requires exclusive legacy semantic context')
if decision_options and not all(decision_values):raise SystemExit('exact semantic decision policy and digest required')
if not 1<=args.semantic_decision_max_calls<=4 or (args.semantic_decision_max_total_input_tokens is not None and args.semantic_decision_max_total_input_tokens<1):raise SystemExit('bounded semantic decision budgets required')
if not decision_options and (args.semantic_decision_max_calls!=3 or args.semantic_decision_max_total_input_tokens is not None):raise SystemExit('semantic decision budgets require an explicit policy')
if decision_options and (args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(finite_values) or any(proof_query_values) or any(contract_values) or any(doctor_values) or semantic_options):raise SystemExit('semantic decision policy requires exclusive bound model coding')
if decision_options and (args.provider!='codex_cli' or args.budget_profile is not None or not 1<=args.timeout<=600 or args.preflight_sleep!=0 or args.semantic_transport_schema!='supervisor-semantic-router-input@1' or args.semantic_metadata_view!='legacy' or args.semantic_working_set!='legacy' or args.semantic_metadata_census_only or args.coding_reply_mode!='legacy' or args.tool_output_policy!='legacy' or args.source_view_advisory_policy!='legacy' or args.repository_source_policy!='legacy' or args.source_input_policy!='legacy' or args.query_reference_policy!='legacy' or args.source_view_profile!='disabled' or args.source_edit_mode!='disabled' or any(source_view_values) or any(formalization_values) or any(numerical_obligation_values) or any(source_lookup_values)):raise SystemExit('semantic decision policy requires exclusive supported Codex model coding')
if args.semantic_context_projection and args.semantic_slot_policy_revision!='closed-typed-identifiers@2':raise SystemExit('semantic context projection requires the versioned identifier policy')
if semantic_options and (args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(finite_values) or any(proof_query_values) or any(contract_values) or any(doctor_values)):raise SystemExit('semantic context options require bound model coding')
if any(value is not None for value in tokenizer_values) and not all(tokenizer_values):raise SystemExit('exact prompt tokenizer descriptor and digest required')
if args.max_input_tokens is not None and args.max_input_tokens<1:raise SystemExit('positive complete prompt token budget required')
if (any(tokenizer_values) or args.max_input_tokens is not None) and (args.preflight or any(finite_values) or any(proof_query_values) or any(contract_values) or any(doctor_values)):raise SystemExit('prompt tokenizer and token budget require router invocation')
if args.source_view_profile!='disabled' and any(value!='legacy' for value in (args.source_view_advisory_policy,args.repository_source_policy,args.source_input_policy,args.query_reference_policy)):raise SystemExit('signed source views cannot combine with advisory or projected source policies')
if not 1<=args.grok_coding_max_turns<=128:raise SystemExit('Grok coding turn bound must be 1 through 128')
if args.grok_coding_max_turns!=128 and (args.provider!='grok_cli' or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or any(contract_values) or any(doctor_values) or any(finite_values) or any(proof_query_values)):raise SystemExit('nondefault coding turns require isolated Grok model-router coding')
if args.source_edit_mode!='disabled' and (args.source_view_profile=='disabled' or args.preflight or args.semantic_metadata_census_only):raise SystemExit('source editing requires selected coding views')
if args.source_view_profile!='disabled' or any(value is not None for value in source_view_values):
 if args.source_view_profile=='disabled' or not all(source_view_values) or args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values):raise SystemExit('exact source descriptor requires isolated semantic coding')
if any(value is not None for value in numerical_obligation_values) and (not all(numerical_obligation_values) or not all(formalization_values) or args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('exact numerical obligation descriptor requires owner formalization coding')
if any(formalization_values) and (not all(formalization_values) or args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('exact owner formalization handoff requires semantic router coding')
if args.semantic_metadata_view!='legacy' and (args.semantic_repository is None or args.purpose!='coding' or args.semantic_transport_schema!='supervisor-semantic-router-input@1' or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('metadata view requires original semantic router coding')
if args.semantic_working_set!='legacy' and (args.source_view_profile=='disabled' or args.semantic_repository is None or args.purpose!='coding' or args.semantic_transport_schema!='supervisor-semantic-router-input@1' or args.semantic_metadata_view!='legacy' or args.semantic_metadata_census_only or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('working set requires original semantic router coding with source retrieval')
if args.semantic_metadata_census_only and args.semantic_metadata_view not in ('common-bindings@1','common-bindings-instruction@1'):raise SystemExit('metadata census requires common bindings')
if args.semantic_transport_schema!='supervisor-semantic-router-input@1' and (args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('controller dictionary transport requires semantic router coding')
if args.tool_output_policy!='legacy' and (args.provider!='codex_cli' or args.semantic_repository is None or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('tool output policy requires semantic Codex router coding')
if args.repository_source_policy!='legacy' and (args.provider!='codex_cli' or args.semantic_repository is None or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or args.source_view_advisory_policy!='legacy' or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('repository source policy requires semantic Codex router coding and exclusive source tools')
if args.source_view_advisory_policy!='legacy' and (args.provider!='codex_cli' or args.semantic_repository is None or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('source view policy requires semantic Codex router coding')
if args.source_input_policy!='legacy' and (args.provider!='codex_cli' or args.semantic_repository is None or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('source input policy requires semantic Codex router coding')
if args.query_reference_policy!='legacy' and (args.provider!='codex_cli' or args.semantic_repository is None or args.purpose!='coding' or args.preflight or args.semantic_metadata_census_only or args.semantic_metadata_view!='legacy' or args.source_input_policy!='legacy' or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('query reference policy requires semantic Codex router coding')
if args.coding_reply_mode!='legacy' and (args.provider!='codex_cli' or args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(contract_values) or any(doctor_values) or any(finite_values)):raise SystemExit('ordinary completion requires semantic Codex router coding')
if any(finite_values) and (not all(finite_values) or any(contract_values) or any(doctor_values) or any(residual_values) or any(instruction_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact finite repository candidate invocation required')
if any(proof_query_values) and (not all(proof_query_values) or not all(finite_values) or any(contract_values) or any(doctor_values) or any(residual_values) or any(instruction_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact finite proof-query context invocation required')
if any(instruction_values) and (not all(instruction_values) or args.preflight or any(contract_values) or any(doctor_values) or args.purpose!='coding'):raise SystemExit('exact public instruction binding requires router coding')
if any(contract_values) and (not all(contract_values) or any(doctor_values) or any(residual_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact local contract candidate invocation required')
if any(doctor_values) and (not all(doctor_values) or args.preflight or args.semantic_repository is not None or any(residual_values)):raise SystemExit('exact Doctor handoff or router invocation required')
if any(residual_values) and (not all(residual_values) or args.preflight or args.semantic_repository is None or args.purpose!='coding'):raise SystemExit('exact Doctor residual context requires semantic coding route')
repository_source_owner_values=(args.repository_source_owner_descriptor,args.repository_source_owner_descriptor_sha256)
if any(value is not None for value in repository_source_owner_values):
 if not all(repository_source_owner_values) or args.repository_source_policy=='legacy' or args.provider!='codex_cli' or args.purpose!='coding' or args.semantic_repository is None or args.preflight or args.semantic_metadata_census_only or args.source_view_advisory_policy!='legacy' or args.source_view_profile!='disabled' or args.source_edit_mode!='disabled' or any(source_view_values) or any(contract_values) or any(doctor_values) or any(finite_values) or any(proof_query_values) or any(residual_values) or decision_options:raise SystemExit('exact repository source owner pair requires exclusive semantic Codex coding')
 repository_source_owner_path=pathlib.Path(args.repository_source_owner_descriptor)
 try:repository_source_owner_path_bytes=len(args.repository_source_owner_descriptor.encode('utf-8'))
 except UnicodeError:raise SystemExit('canonical repository source owner descriptor path required')
 if not 1<=repository_source_owner_path_bytes<=4096 or not repository_source_owner_path.is_absolute() or str(repository_source_owner_path)!=args.repository_source_owner_descriptor or args.repository_source_owner_descriptor.startswith('//') or str(repository_source_owner_path)=='/' or '..' in repository_source_owner_path.parts or any(ord(char)<32 or ord(char)==127 or char=='\\' for char in args.repository_source_owner_descriptor):raise SystemExit('canonical repository source owner descriptor path required')
 if len(args.repository_source_owner_descriptor_sha256)!=64 or any(char not in '0123456789abcdef' for char in args.repository_source_owner_descriptor_sha256):raise SystemExit('exact repository source owner descriptor SHA-256 required')
"""
    + _REPOSITORY_SOURCE_ISOLATION_GUARD
    + r"""if args.repository_source_policy!='legacy':raise SystemExit('repository_source_owner_transport_unavailable')
if args.repository_source_policy!='legacy' and not all(repository_source_owner_values):raise SystemExit('repository source policy requires its owner descriptor pair')
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router_runtime
# Earlier archives have no shared constant and retain the original worker cap.
worker_timeout_limit=getattr(router_runtime,'MAX_IMPLEMENTATION_TIMEOUT_SECONDS',300)
if type(worker_timeout_limit)is not int or not 1<=worker_timeout_limit<=600:raise SystemExit('invalid runtime timeout ceiling')
if args.budget_profile is not None:
 worker_timeout_limit=router_runtime.implementation_timeout_limit(args.budget_profile)
 if type(worker_timeout_limit)is not int or not 1<=worker_timeout_limit<=1800:raise SystemExit('invalid selected timeout ceiling')
if not 1<=args.timeout<=worker_timeout_limit or not 0<=args.preflight_sleep<=30:raise SystemExit('bounded timeout required')
if args.preflight:
 prompt=sys.stdin.buffer.read(2*1024*1024+1)
 if len(prompt)>2*1024*1024:raise SystemExit('preflight prompt exceeds bound')
 (pathlib.Path.cwd()/'worker-preflight-prompt.txt').write_bytes(prompt)
 target=pathlib.Path.cwd()/'worker-preflight.txt';target.write_text('worker candidate edit\n')
 denied=[]
 for private in (root/'state/preflight-key',root/'state/preflight-owner.duckdb'):
  for mode in ('rb','ab'):
   try:
    with private.open(mode):pass
   except PermissionError:denied.append(str(private)+':'+mode)
   else:raise SystemExit('private owner authority accessible')
 marker=pathlib.Path.cwd()/'.git'
 for action in ('write','unlink'):
  try:
   if action=='write':marker.open('ab').close()
   else:marker.unlink()
  except PermissionError:denied.append('.git:'+action)
  else:raise SystemExit('worker can mutate Git worktree authority')
 import subprocess
 top=subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()
 if pathlib.Path(top)!=pathlib.Path.cwd():raise SystemExit('worker cannot read exact Git worktree identity')
 if args.preflight_sleep:subprocess.Popen(['/bin/sleep','30'],start_new_session=True)
 print(json.dumps({'schema':'container-worker-preflight@1','pid':os.getpid(),'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),'boundary':evidence,'candidate_write':target.read_text()=='worker candidate edit\n','private_denials':denied,'workspace':str(pathlib.Path.cwd()),'prompt_bytes':len(prompt),'prompt_sha256':hashlib.sha256(prompt).hexdigest(),'original_task_python':subprocess.check_output(['python3','--version'],text=True).strip(),'runtime_python':sys.version.split()[0],'provider_calls':0}),flush=True)
 if args.preflight_sleep:time.sleep(args.preflight_sleep)
elif args.finite_proof_query_context is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_context import main
 sys.argv=[sys.argv[0],'--artifact',str(args.finite_repository_artifact),'--sha256',args.finite_repository_sha256,'--task-cid',args.finite_repository_task_cid,'--context',str(args.finite_proof_query_context),'--context-sha256',args.finite_proof_query_sha256,'--context-cid',args.finite_proof_query_context_cid]
 raise SystemExit(main())
elif args.finite_repository_artifact is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner import main
 sys.argv=[sys.argv[0],'--artifact',str(args.finite_repository_artifact),'--sha256',args.finite_repository_sha256,'--task-cid',args.finite_repository_task_cid]
 raise SystemExit(main())
elif args.doctor_contract_artifact is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import main
 sys.argv=[sys.argv[0],'--artifact',str(args.doctor_contract_artifact),'--sha256',args.doctor_contract_sha256,'--task-cid',args.doctor_contract_task_cid]
 raise SystemExit(main())
elif args.doctor_candidate_artifact is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import main
 sys.argv=[sys.argv[0],'--artifact',str(args.doctor_candidate_artifact),'--sha256',args.doctor_candidate_sha256,'--task-cid',args.doctor_task_cid]
 raise SystemExit(main())
else:
 if decision_options:
  from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker import main
 else:
  from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main
 from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper import run_with_child_custody
 sys.argv=[sys.argv[0],'--provider',args.provider,'--model',args.model,'--reasoning-effort',args.reasoning_effort,'--purpose',args.purpose,'--timeout',str(args.timeout),'--max-output-tokens',str(args.max_output_tokens),'--container-boundary',str(artifact),'--container-boundary-sha256',digest]
 if args.budget_profile is not None:sys.argv+=['--budget-profile',args.budget_profile]
 if args.semantic_repository is not None:sys.argv+=['--semantic-repository',str(args.semantic_repository)]
 if args.semantic_slot_policy_revision!='legacy@1':sys.argv+=['--semantic-slot-policy-revision',args.semantic_slot_policy_revision]
 if args.semantic_context_projection:sys.argv+=['--semantic-context-projection']
 if args.semantic_native_input:sys.argv+=['--semantic-native-input']
 if decision_options:
  sys.argv+=['--semantic-decision-policy-artifact',str(args.semantic_decision_policy_artifact),'--semantic-decision-policy-sha256',args.semantic_decision_policy_sha256,'--semantic-decision-max-calls',str(args.semantic_decision_max_calls)]
  if args.semantic_decision_max_total_input_tokens is not None:sys.argv+=['--semantic-decision-max-total-input-tokens',str(args.semantic_decision_max_total_input_tokens)]
 if args.prompt_tokenizer_artifact is not None:sys.argv+=['--prompt-tokenizer-artifact',str(args.prompt_tokenizer_artifact),'--prompt-tokenizer-sha256',args.prompt_tokenizer_sha256]
 if args.max_input_tokens is not None:sys.argv+=['--max-input-tokens',str(args.max_input_tokens)]
 if args.semantic_transport_schema!='supervisor-semantic-router-input@1':sys.argv+=['--semantic-transport-schema',args.semantic_transport_schema]
 if args.semantic_metadata_view!='legacy':sys.argv+=['--semantic-metadata-view',args.semantic_metadata_view]
 if args.semantic_working_set!='legacy':sys.argv+=['--semantic-working-set',args.semantic_working_set]
 if args.semantic_metadata_census_only:sys.argv+=['--semantic-metadata-census-only']
 if args.coding_reply_mode!='legacy':sys.argv+=['--coding-reply-mode',args.coding_reply_mode]
 if args.tool_output_policy!='legacy':sys.argv+=['--tool-output-policy',args.tool_output_policy]
 if args.source_view_advisory_policy!='legacy':sys.argv+=['--source-view-advisory-policy',args.source_view_advisory_policy]
 if args.repository_source_policy!='legacy':sys.argv+=['--repository-source-policy',args.repository_source_policy]
 if args.repository_source_owner_descriptor is not None:sys.argv+=['--repository-source-owner-descriptor',args.repository_source_owner_descriptor,'--repository-source-owner-descriptor-sha256',args.repository_source_owner_descriptor_sha256]
 if args.source_input_policy!='legacy':sys.argv+=['--source-input-policy',args.source_input_policy]
 if args.query_reference_policy!='legacy':sys.argv+=['--query-reference-policy',args.query_reference_policy]
 if args.doctor_residual_artifact is not None:sys.argv+=['--doctor-residual-artifact',str(args.doctor_residual_artifact),'--doctor-residual-sha256',args.doctor_residual_sha256,'--doctor-residual-task-cid',args.doctor_residual_task_cid]
 if args.public_instruction_artifact is not None:sys.argv+=['--public-instruction-artifact',str(args.public_instruction_artifact),'--public-instruction-sha256',args.public_instruction_sha256,'--public-instruction-task-cid',args.public_instruction_task_cid]
 if args.source_formalization_handoff is not None:sys.argv+=['--source-formalization-handoff',str(args.source_formalization_handoff),'--source-formalization-handoff-sha256',args.source_formalization_handoff_sha256]
 if args.numerical_obligation_descriptor is not None:sys.argv+=['--numerical-obligation-descriptor',str(args.numerical_obligation_descriptor),'--numerical-obligation-descriptor-sha256',args.numerical_obligation_descriptor_sha256]
 if args.source_lookup_descriptor is not None:sys.argv+=['--source-lookup-descriptor',str(args.source_lookup_descriptor),'--source-lookup-descriptor-sha256',args.source_lookup_descriptor_sha256]
 if args.grok_coding_max_turns!=128:sys.argv+=['--grok-coding-max-turns',str(args.grok_coding_max_turns)]
 if args.source_view_profile!='disabled':sys.argv+=['--source-view-profile',args.source_view_profile,'--source-view-descriptor',str(args.source_view_descriptor),'--source-view-descriptor-sha256',args.source_view_descriptor_sha256]
 if args.source_edit_mode!='disabled':sys.argv+=['--source-edit-mode',args.source_edit_mode]
 raise SystemExit(run_with_child_custody(main))
""")


async def install_worker_boundary(environment, *, container_name: str, output: Path, provider: str = "codex_cli",
                                  repository_source_resources: dict | None = None) -> dict:
    if provider not in {"codex_cli", "grok_cli"}:
        raise ValueError("explicit supported container provider required")
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_configuration import validate_source_owner_configuration
    repository_source_resources = validate_source_owner_configuration(repository_source_resources,
        owner_private_paths=[ROOT + "/state"])
    if repository_source_resources is not None and provider != "codex_cli":
        raise ValueError("repository source resources require the Codex worker boundary")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    async def run(label, command, *, user="root", env=None, timeout=180):
        result = await environment.exec(
            command=command, cwd="/app", user=user, env=env, timeout_sec=timeout
        )
        (output / (label + ".log")).write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code:
            raise RuntimeError("worker deployment " + label + " failed")
        return result

    # Select only nonsecret engine identity fields, never Docker's environment.
    inspect = subprocess.check_output(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .Id}} {{json .Image}} {{json .State.Pid}} {{json .State.Running}} {{json .HostConfig.Privileged}}",
            container_name,
        ],
        text=True,
    ).split()
    cid, image, pid, running, privileged = [json.loads(x) for x in inspect]
    if not running or privileged:
        raise ValueError("running ordinary Docker task container required")
    namespaces = json.loads(
        (
            await run(
                "namespaces",
                "python3 -I -c "
                + shlex.quote(
                    "import json,os;print(json.dumps({k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')}))"
                ),
            )
        ).stdout
    )
    host_namespaces = {
        k: __import__("os").readlink(f"/proc/{pid}/ns/{k}") for k in ("pid", "mnt", "net")
    }
    if host_namespaces != namespaces:
        raise ValueError("Docker engine and observed namespace identities differ")
    provider_home = ".grok" if provider == "grok_cli" else ".codex"
    auth_home = "grok-auth" if provider == "grok_cli" else "codex-auth"
    await run(
        "worker-account",
        f"apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo && "
        f"useradd --uid 1001 --gid supervisor --home-dir {ROOT}/worker-home --shell /usr/sbin/nologin benchmarkworker && "
        f"install -d -o root -g root -m 0755 {ROOT}/bin {ROOT}/provider-bin && "
        f"install -d -o supervisor -g supervisor -m 0750 {ROOT}/worktrees && "
        f"install -d -o supervisor -g supervisor -m 0750 {ROOT}/worker-observations && "
        f"install -d -o benchmarkworker -g supervisor -m 0700 {ROOT}/worker-home {ROOT}/worker-home/{provider_home} && "
        f"install -o benchmarkworker -g supervisor -m 0600 {ROOT}/{auth_home}/auth.json {ROOT}/worker-home/{provider_home}/auth.json && "
        f"chmod 0700 {ROOT}/home {ROOT}/{auth_home} {ROOT}/state",
    )
    if provider == "codex_cli":
        await run("native-codex-binary", "python3 -I -c " + shlex.quote(native_codex_exposure_script()))
    else:
        # Deployment already exposed and pinned this root-owned executable.
        from .terminal_grok_deployment import require_grok_version_output
        observed = await run("native-grok-version", ROOT + "/provider-bin/grok --version")
        require_grok_version_output(observed.stdout)
    boundary = {
        "schema": "supervisor-container-worker-boundary@1",
        "container_id": cid,
        "image_id": image,
        "namespaces": namespaces,
        "owner_uid": 1000,
        "worker_uid": 1001,
        "allowed_worktree_roots": [ROOT + "/worktrees"],
        "owner_private_paths": [ROOT + "/state", ROOT + "/home", ROOT + "/" + auth_home],
        "provider": provider,
        "validation_repository_roots": ["/app"],
        "single_worker": True,
    }
    if repository_source_resources is not None:
        boundary["repository_source_resources"] = repository_source_resources
    raw = (json.dumps(boundary, sort_keys=True, indent=2) + "\n").encode()
    sudoers = (
        "Defaults:supervisor !use_pty\nsupervisor ALL=(benchmarkworker) NOPASSWD: "
        + ROOT
        + "/bin/worker-entry *\n"
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name, data, target, mode in [
            ("boundary", raw, ROOT + "/container-boundary.json", "0444"),
            ("owner", OWNER_ENTRY.encode(), ROOT + "/bin/router-worker", "0555"),
            ("validation", VALIDATION_ENTRY.encode(), ROOT + "/bin/validation-worker", "0555"),
            ("worker", WORKER_ENTRY.encode(), ROOT + "/bin/worker-entry", "0555"),
            ("source-lookup", SOURCE_LOOKUP_ENTRY.encode(), ROOT + "/bin/source-lookup", "0555"),
            ("sudoers", sudoers.encode(), "/etc/sudoers.d/supervisor-worker", "0440"),
        ]:
            path = Path(temporary) / name
            path.write_bytes(data)
            await environment.upload_file(path, target)
            await run(
                "protect-" + name, "chown root:root " + target + " && chmod " + mode + " " + target
            )
    await run("sudoers-check", "visudo -cf /etc/sudoers.d/supervisor-worker")
    await run(
        "canonical-git-protect",
        "find /app/.git -type d -exec chmod 0750 {} + && find /app/.git -type f -exec chmod 0640 {} +",
    )
    result = {
        "schema": "container-worker-deployment@1",
        "boundary": boundary,
        "boundary_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_command": ROOT
        + "/bin/router-worker --provider " + provider + " --model "
        + ("grok-4.7" if provider == "grok_cli" else "gpt-6.1-sol") + " --reasoning-effort high --timeout 90",
        "candidate_runner_argv": [ROOT + "/bin/validation-worker"],
        "worker_worktree_root": ROOT + "/worktrees",
        "provider_calls": 0,
        "qualified": False,
    }
    (output / "installation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


async def deploy_worker_boundary(environment, *, output: Path, provider: str = "codex_cli",
                                 repository_source_resources: dict | None = None) -> dict:
    """Fresh Harbor deployment: install and exercise the actual worker boundary."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_configuration import validate_source_owner_configuration
    repository_source_resources = validate_source_owner_configuration(
        repository_source_resources, owner_private_paths=[ROOT + "/state"])
    if repository_source_resources is not None and provider != "codex_cli":
        raise ValueError("repository source resources require the Codex worker route")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Harbor's own service lookup resolves the original running task container;
    # the installer independently compares engine and in-container namespaces.
    container = await environment._platform._resolve_service_container("main")
    installed = await install_worker_boundary(
        environment, container_name=container, output=output / "installation", provider=provider,
        **({"repository_source_resources": repository_source_resources} if repository_source_resources is not None else {})
    )
    qualified = await qualify_worker_boundary(environment, output=output / "qualification")
    result = {**installed, "qualified": qualified["qualified"], "qualification": qualified}
    (output / "deployment.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


async def qualify_worker_boundary(environment, *, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    probe = r"""import json,os,pathlib,signal,subprocess,time
root=pathlib.Path('/opt/ipfs-supervisor');state=root/'state';candidate=root/'worktrees/provider-free'
subprocess.run(['git','-C','/app','worktree','add','--detach',str(candidate),'HEAD'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
(state/'preflight-key').write_text('synthetic-noncredential-private-check\n');(state/'preflight-key').chmod(0o600)
(state/'preflight-owner.duckdb').write_bytes(b'synthetic-owner-state');(state/'preflight-owner.duckdb').chmod(0o600)
command=[str(root/'bin/router-worker'),'--preflight']
completed=subprocess.run(command,cwd=candidate,text=True,capture_output=True,timeout=30)
if completed.returncode:raise RuntimeError(completed.stderr)
evidence=json.loads(completed.stdout)
if not evidence['candidate_write'] or len(evidence['private_denials'])!=6:raise ValueError('worker permission preflight failed')
if (candidate/'worker-preflight.txt').stat().st_uid!=1001:raise ValueError('actual worker did not write candidate')
# A second preflight has no caller-controlled code. Terminate its actual sudo
# monitor and require both monitor and exact observed worker process to exit.
(candidate/'worker-preflight.txt').unlink()
process=subprocess.Popen(command+['--preflight-sleep','30'],cwd=candidate,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
sleeping=json.loads(process.stdout.readline());worker=sleeping['pid']
os.kill(process.pid,signal.SIGTERM)
try:process.wait(timeout=10)
except subprocess.TimeoutExpired:
 os.killpg(process.pid,signal.SIGKILL);process.wait();raise
for _ in range(50):
 try:
  status=pathlib.Path('/proc')/str(worker)/'stat'
  if not status.exists() or status.read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise ValueError('worker survived owner termination')
remaining=[]
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  status=(path/'status').read_text();uid=next(x for x in status.splitlines() if x.startswith('Uid:')).split()[1]
  state_line=next(x for x in status.splitlines() if x.startswith('State:'))
  if uid=='1001' and 'Z' not in state_line:remaining.append(int(path.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):pass
if remaining:raise ValueError('live worker processes remain')
subprocess.run(['git','-C','/app','worktree','remove','--force',str(candidate)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
print(json.dumps({'qualified':True,'permission_evidence':evidence,'termination_worker_pid':worker,'remaining_worker_pids':remaining,'provider_calls':0,'completion_authority':False}))
"""
    result = await environment.exec(
        command=PYTHON + " -I -c " + shlex.quote(probe),
        cwd="/app",
        user="supervisor",
        env=runtime_environment(),
        timeout_sec=60,
    )
    (output / "probe.log").write_text((result.stdout or "") + (result.stderr or ""))
    if result.return_code:
        raise RuntimeError("provider-free worker qualification failed")
    evidence = json.loads(result.stdout)
    (output / "qualification.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence
