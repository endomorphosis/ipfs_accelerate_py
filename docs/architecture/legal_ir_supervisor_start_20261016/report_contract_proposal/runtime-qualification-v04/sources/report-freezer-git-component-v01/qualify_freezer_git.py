"""Genuine finite Git component mechanics; never public freezer authority.

Root executes only after reviewing the exact request. This selects the actual
source-patched original lock and shared commit-tail AST. It never creates a
provider outcome, dispatch capture, fake native owner, daemon facade or active
H context and never calls freeze/_freeze_locked. No task state/backend changes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import threading

HERE=Path(__file__).absolute().parent
NEW=HERE.parent
SOURCE=NEW/'report-freezer-v02'
FREEZER_SHA='2e88a05782c96aed0f64583f414209a3535502c034b7a8f610f8577d60ae345e'
RUN=Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-report-freezer-git-component-20261017-r01')
D=Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
A=Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
DATASETS_ROOT=Path('/home/barberb/lift_coding/external/ipfs_datasets')
ROOTS=(DATASETS_ROOT/'workspace/test-logs',DATASETS_ROOT/'workspace/todo-queues',DATASETS_ROOT/'docs/implementation/reports/evidence',Path('/tmp/pytest-of-barberb'))
STATE=Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY='f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
LEDGER=DATASETS_ROOT/'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
BOARD_SOURCE=Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/tasks.todo.md')
BOARD='docs/architecture/legal_ir_supervisor_start_20261016/tasks.todo.md'
FALSE={'provider_calls_performed':False,'actual_provider_outcome_observed':False,'public_report_freezer_qualified':False,
       'active_success_callback_minted':False,'actual_provider_execution_qualified':False,'production_hook_deployed':False,
       'native_validation_qualified':False,'acceptance_authority':False,'completion_authority':False,
       'canonical_tasks_accepted':0,'canonical_attempt_counters_changed':False}


def need(value,reason):
    if not value:raise ValueError(reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(path,maximum=16*1024*1024):
    path=Path(path)
    need(path.is_absolute() and str(path)==os.path.abspath(path) and not any(p.is_symlink() for p in (path,*path.parents)),
         'canonical_source_required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_uid in {0,os.getuid()}
             and before.st_mode & 0o002==0 and 0<before.st_size<=maximum,'bounded_owned_regular_source_required')
        raw=stream.read(maximum+1);after=os.fstat(stream.fileno())
    witness=lambda m:(m.st_dev,m.st_ino,m.st_size,m.st_mode,m.st_uid,m.st_gid,m.st_nlink,m.st_mtime_ns,m.st_ctime_ns)
    need(len(raw)==before.st_size and witness(before)==witness(after)==witness(path.lstat()),'stable_source_required')
    return raw


def strict(raw):
    def pairs(items):
        result={}
        for k,v in items:
            need(k not in result,'duplicate_json_key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite_json')))


def wire(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_new(path,raw):
    need(type(raw) is bytes and 0<len(raw)<=8*1024*1024,'bounded_private_output_required')
    path=Path(path)
    need(RUN in path.parents and not any(p.is_symlink() for p in (path,*path.parents)),'only_fresh_fixture_output_allowed')
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as stream:
        os.fchmod(stream.fileno(),0o644);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fsync_dir(path.parent)
    return path.lstat().st_dev,path.lstat().st_ino


def remove_original_fixture_file(path,identity):
    meta=path.lstat()
    need(RUN in path.parents and (meta.st_dev,meta.st_ino)==identity and meta.st_uid==os.getuid()
         and stat.S_ISREG(meta.st_mode) and meta.st_nlink==1,'fixture_file_identity_changed')
    path.unlink();fsync_dir(path.parent)


def guard(request):
    need(type(request.get('source_sha256')) is dict and 20<=len(request['source_sha256'])<=128,'finite_source_closure_required')
    for spelling,digest in request['source_sha256'].items():
        need(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest) and sha(read(Path(spelling)))==digest,
             'frozen_source_closure_changed')
    native=D/'ipfs_datasets_py/optimizers/logic_theorem_optimizer'
    required={str(Path(__file__).absolute()),str(HERE/'SOURCE_MANIFEST.json'),str(HERE/'README.md'),
              str(SOURCE/'SOURCE_MANIFEST.json'),str(BOARD_SOURCE),'/usr/bin/python3.12','/usr/bin/git'}
    required.update(str(native/(name+'.py')) for name in ('parented_phase_resources','autoencoder_daemon_resources',
        'existing_resource_scheduler','resource_scheduler','runtime_telemetry','proof_resource_safety',
        'validation_run_resources','autoencoder_resource_recovery'))
    required.update(str(D/name) for name in ('ipfs_datasets_py/__init__.py','ipfs_datasets_py/router_deps.py',
        'ipfs_datasets_py/optimizers/__init__.py','ipfs_datasets_py/optimizers/logic_theorem_optimizer/__init__.py'))
    need(required<=set(request['source_sha256']),'mandatory_native_toolchain_component_source_closure_required')
    own=read(HERE/'SOURCE_MANIFEST.json')
    need(sha(own)==request.get('qualifier_source_manifest_sha256'),'actual_qualifier_source_manifest_changed')
    raw=read(SOURCE/'SOURCE_MANIFEST.json')
    need(sha(raw)==FREEZER_SHA,'frozen_freezer_manifest_changed')
    source=strict(raw)
    for row in source['files']:
        need(request['source_sha256'].get(str(SOURCE/row['path']))==row['sha256'],'complete_freezer_rows_required')
    need(all(request['source_sha256'].get(path)==digest for path,digest in source['input_pins'].items()),
         'complete_reviewed_input_pins_required')
    need(request.get('freezer_source_manifest_sha256')==FREEZER_SHA,'exact_freezer_generation_required')


def module_from_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def derive_components(owner):
    # The assembler is source-only and exact-baseline guarded. It applies no
    # patch to owned sources. Only the small actual lock successor is imported.
    builder=module_from_path('freezer_component_source_assembler',SOURCE/'build_patch.py')
    original_lock=read(A/builder.LOCK).decode();original_daemon=read(A/builder.DAEMON).decode()
    need(sha(original_lock.encode())==builder.EXPECTED_LOCK and sha(original_daemon.encode())==builder.EXPECTED_DAEMON,
         'actual_original_component_baseline_changed')
    lock_source=builder.lock_successor(original_lock)
    daemon_source=builder.daemon_successor(original_daemon)
    baseline=strict(read(SOURCE/'BASELINE.json'))
    wanted={Path(row['path']).name:row['proposed_sha256'] for row in baseline['inputs']}
    need(sha(lock_source.encode())==wanted['git_checkout_transaction.py']
         and sha(daemon_source.encode())==wanted['implementation_daemon.py'],'source_patched_successor_changed')
    lock_path=owner.directory/'component_sources/git_checkout_transaction.py'
    write_new(lock_path,lock_source.encode())
    lock=module_from_path('actual_source_patched_checkout_component',lock_path)
    tree=ast.parse(daemon_source)
    daemon=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PortalImplementationDaemon')
    tail=next(n for n in daemon.body if isinstance(n,ast.FunctionDef) and n.name=='_commit_staged_worktree_changes')
    tail_ast=ast.dump(tail,include_attributes=False)
    selected='from __future__ import annotations\n\nclass OriginalSharedTailComponent:\n'+''.join(
        '    '+line+'\n' for line in ast.unparse(tail).splitlines())
    selected_path=owner.directory/'component_sources/original_shared_tail.py'
    write_new(selected_path,selected.encode())
    tail_module=module_from_path('actual_source_patched_shared_tail_component',selected_path)
    helper=module_from_path('actual_frozen_report_freezer_git_component',SOURCE/'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py')
    return lock,tail_module.OriginalSharedTailComponent(),helper,{
        'lock_successor_sha256':sha(lock_source.encode()),'daemon_successor_sha256':sha(daemon_source.encode()),
        'selected_original_shared_tail_AST_sha256':sha(tail_ast.encode()),'selected_component_source_sha256':sha(selected.encode()),
        'selection_scope':'actual shared-tail AST only; full daemon/provider/freezer never instantiated'}


@dataclass(frozen=True)
class ComponentTask:
    task_id:str='FIXTURE-REPORT-COMPONENT'
    title:str='Synthetic nonmodel report commit mechanics only'


def make_repository(owner,helper,tail,name):
    repo=owner.directory/name;repo.mkdir(mode=0o700);fsync_dir(owner.directory)
    owner.account_external_bytes(helper.storage_charge_key(repo),128*1024*1024)
    git=helper._Git(owner,repo)
    git.run(['init','-b','fixture-report-component'])
    original=read(BOARD_SOURCE)
    need(sha(original)==helper.BOARD_SHA256,'exact_normative_fixture_board_changed')
    write_new(repo/BOARD,original)
    git.run(['add','--',BOARD])
    result=tail._commit_staged_worktree_changes(repo,ComponentTask(),0,status='explicit metadata baseline',git_runner=git.commit_runner)
    baseline=result['commit'];need(re.fullmatch('[0-9a-f]{40}',baseline),'actual_fixture_baseline_commit_required')
    # Ignore only the declared report's private directory using local fixture
    # metadata, preserving the baseline tree as board-only.
    write_new(repo/'.git/info/exclude-fixture-note',b'This separate file is private fixture metadata.\n')
    exclude=repo/'.git/info/exclude'
    existing=helper._read_git_metadata(exclude,65536)
    with exclude.open('ab') as stream:stream.write(b'\ndata/\n');stream.flush();os.fsync(stream.fileno())
    need(read(exclude).startswith(existing),'actual_fixture_exclude_preimage_changed')
    fsync_dir(exclude.parent)
    return repo,git,baseline


def expected_refusal(observations,name,exception,operation,reason=None):
    try:operation()
    except exception as error:
        actual=getattr(error,'reason',str(error))
        need(reason is None or actual==reason,'unexpected_fixture_refusal:'+str(actual))
        observations.append({'case':name,'passed':True,'actual_refusal':str(actual)})
    else:raise ValueError('expected_fixture_refusal_missing:'+name)


def component(owner,request):
    lock,tail,h,derived=derive_components(owner)
    observations=[]
    repo,git,baseline=make_repository(owner,h,tail,'positive-repository')
    declared={str(repo):128*1024*1024}
    h._check_storage(owner,repo,repo/'.git',repo/'.git',declared)
    report_raw=wire({'schema':'legal-ir-explicit-report-freezer-git-fixture@1',
        'scope':'synthetic nonmodel Git component; no actual reconciliation/provider/acceptance',**FALSE})
    write_new(repo/h.REPORT,report_raw)
    with lock.checkout_git_directory_mutation(repo/'.git') as actual_directory:
        need(actual_directory==repo/'.git','actual_lock_directory_changed')
        with lock.checkout_git_directory_mutation(repo/'.git') as nested:
            need(nested==actual_directory,'original_reentrant_registry_not_shared')
        observations.append({'case':'actual_factored_original_lock_reentrant','passed':True})
        need(git.run(['rev-parse','--verify','HEAD^{commit}']).decode().strip()==baseline,'actual_fixture_baseline_drifted')
        need(git.run(['diff','--cached','--name-only','-z',baseline,'--'])==b'','actual_fixture_index_not_clean')
        need(git.run(['check-ignore','--no-index','--',h.REPORT]).decode().strip()==h.REPORT,'fixture_report_not_ignored')
        initial=h._index_snapshot(repo/'.git')
        h.strict_report(h.read_regular(repo/h.REPORT))
        need(h._index_snapshot(repo/'.git')==initial,'actual_fixture_index_preimage_changed')
        git.run(['add','--sparse','-f','--',h.REPORT])
        staged=h.parse_diff(git.run(['diff','--cached','--raw','--abbrev=40','--no-renames','--no-ext-diff','-z',baseline,'--']))
        need(git.run(['cat-file','blob',staged],h.MAX_DOCUMENT)==report_raw,'actual_staged_report_bytes_differ')
        index=h._index_snapshot(repo/'.git')
        result=tail._commit_staged_worktree_changes(repo,ComponentTask(),0,status='synthetic report-only candidate',git_runner=git.commit_runner)
        candidate=result['commit']
        tree=h.parse_commit(git.run(['cat-file','commit',candidate]),baseline)
        need(h.parse_diff(git.run(['diff-tree','--no-commit-id','--raw','--abbrev=40','--no-renames','--no-ext-diff','-r','-z',baseline,candidate,'--']))==staged,
             'actual_fixture_commit_extra_path')
        board_oid=h.parse_tree(git.run(['ls-tree','-z',candidate,'--',BOARD]),BOARD)
        need(git.run(['cat-file','blob',board_oid],h.MAX_DOCUMENT)==read(BOARD_SOURCE),'actual_committed_board_changed')
        need(h.parse_tree(git.run(['ls-tree','-z',candidate,'--',h.REPORT]),h.REPORT)==staged,'actual_committed_report_oid_changed')
        need(git.run(['status','--porcelain=v1','-z','--untracked-files=all'])==b'','actual_fixture_candidate_dirty')
        need(git.run(['rev-parse','--verify','HEAD^{commit}']).decode().strip()==candidate,'actual_fixture_candidate_not_head')
        observations.append({'case':'actual_ignored_report_forced_staged_and_original_tail_committed','passed':True,
            'baseline_commit':baseline,'candidate_commit':candidate,'candidate_parent':baseline,'candidate_tree':tree,
            'report_sha256':sha(report_raw),'board_sha256':h.BOARD_SHA256,'actual_git_children':git.count,
            'public_provider_freezer_exercised':False})
    # The original guard must preserve known fixture-owned foreign/custody files.
    index_before=h._index_snapshot(repo/'.git')
    for filename,reason,label in [('index.lock','checkout_git_foreign_lock_present','foreign_index_lock_preserved'),
                                 (lock.PENDING_CUSTODY_NAME,'checkout_git_retained_contention_requires_recovery','pending_custody_marker_preserved')]:
        path=repo/'.git'/filename;raw=b'{"explicit_fixture":true}\n';identity=write_new(path,raw)
        def enter_lock():
            with lock.checkout_git_directory_mutation(repo/'.git'):raise ValueError('foreign_guard_entered')
        expected_refusal(observations,label,lock.CheckoutGitMutationBlocked,enter_lock,reason)
        need(read(path)==raw and h._index_snapshot(repo/'.git')==index_before,'original_guard_mutated_foreign_fixture_file_or_index')
        remove_original_fixture_file(path,identity)
    # Actual unexpected paths are observed and rejected; only fixture-owned
    # files are removed, never any Git lock or index owned by another worker.
    for relative,args,label in [('unexpected.txt',['ls-files','--others','--exclude-standard','-z'],'untracked_extra_path_refused'),
                                ('data/unexpected.json',['ls-files','--others','--ignored','--exclude-standard','-z'],'ignored_extra_path_refused')]:
        path=repo/relative;identity=write_new(path,b'explicit negative fixture\n')
        raw=git.run(args)
        expected_refusal(observations,label,h.ReportFreezerRefusal,lambda:h.allowed_dirty_paths(raw),'extra_dirty_path_refused')
        remove_original_fixture_file(path,identity)
    wrong_parent='0'*40 if baseline!='0'*40 else 'f'*40
    commit_raw=git.run(['cat-file','commit',candidate])
    expected_refusal(observations,'actual_commit_wrong_baseline_parent_refused',h.ReportFreezerRefusal,
        lambda:h.parse_commit(commit_raw,wrong_parent),'exact_single_baseline_parent_required')
    actual_head=git.run(['rev-parse','--verify','HEAD^{commit}']).decode().strip()
    expected_refusal(observations,'existing_candidate_is_not_original_baseline',h.ReportFreezerRefusal,
        lambda:h.need(actual_head==baseline,'provider_created_commit_or_baseline_drift_refused'),
        'provider_created_commit_or_baseline_drift_refused')
    # Use a separate fresh real repository for a staged-extra negative. Leave
    # that private staged index untouched as evidence; never reset or rescue.
    staged_repo,other_git,other_baseline=make_repository(owner,h,tail,'staged-extra-repository')
    write_new(staged_repo/'unexpected.txt',b'explicit staged negative fixture\n')
    other_git.run(['add','--','unexpected.txt'])
    diff=other_git.run(['diff','--cached','--raw','--abbrev=40','--no-renames','--no-ext-diff','-z',other_baseline,'--'])
    expected_refusal(observations,'actual_extra_staged_path_refused',h.ReportFreezerRefusal,lambda:h.parse_diff(diff),'report_only_diff_required')
    # Modify only our separate fixture report and index, and preserve both
    # failed preimages. No further Git command reads this deliberately changed
    # private index and no canonical source is touched.
    before_report=h.read_regular(repo/h.REPORT)
    with (repo/h.REPORT).open('ab') as stream:stream.write(b' ');stream.flush();os.fsync(stream.fileno())
    expected_refusal(observations,'actual_report_preimage_drift_refused',h.ReportFreezerRefusal,
        lambda:h.need(h.read_regular(repo/h.REPORT)==before_report,'report_or_staged_preimage_changed'),'report_or_staged_preimage_changed')
    before_index=h._index_snapshot(staged_repo/'.git')
    with (staged_repo/'.git/index').open('ab') as stream:stream.write(b'EXPLICIT-FIXTURE-INDEX-DRIFT');stream.flush();os.fsync(stream.fileno())
    expected_refusal(observations,'actual_index_preimage_drift_refused',h.ReportFreezerRefusal,
        lambda:h.need(h._index_snapshot(staged_repo/'.git')==before_index,'staged_commit_preimage_changed'),'staged_commit_preimage_changed')
    need(len(observations)==11 and git.count+other_git.count==29,'finite_actual_component_case_or_git_command_count_differs')
    owner.check_usage(owner.directory,full=True)
    return {'schema':'legal-ir-report-freezer-git-component@1','passed':all(row['passed'] for row in observations),
        'scope':'synthetic genuine Git/lock/shared-tail mechanics only','cases':observations,'controlled_cases':11,
        'actual_git_children':git.count+other_git.count,'actual_four_slot_native_owner':True,'derived_source_components':derived,
        'freezer_source_manifest_sha256':FREEZER_SHA,'source_request_sha256':request['_actual_request_sha256'],**FALSE}


def fsync_tree(owner):
    count=0
    for directory,subdirs,files in os.walk(owner.directory,followlinks=False):
        for name in [*subdirs,*files]:
            path=Path(directory)/name;meta=path.lstat();count+=1
            need(count<=4096 and not stat.S_ISLNK(meta.st_mode),'finite_regular_fixture_durability_tree_required')
            if stat.S_ISREG(meta.st_mode):
                fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                try:os.fsync(fd)
                finally:os.close(fd)
        fsync_dir(Path(directory))
    need(owner.check_usage(owner.directory,full=True)['group_rss']['live_processes']==0,'actual_git_children_not_quiescent')


def main():
    need(len(sys.argv)==3 and re.fullmatch('[0-9a-f]{64}',sys.argv[2]),'exact_request_path_and_sha_required')
    raw=read(Path(sys.argv[1]).absolute());need(sha(raw)==sys.argv[2],'reviewed_request_changed')
    request=strict(raw)
    need(request.get('schema')=='legal-ir-report-freezer-git-component-request@1'
         and request.get('run_root')==str(RUN) and request.get('scheduler_state_path')==str(STATE)
         and request.get('expected_scheduler_policy_sha256')==POLICY and request.get('ledger_path')==str(LEDGER)
         and request.get('roots')==[str(p) for p in ROOTS],'exact_native_authority_and_fresh_fixture_required')
    guard(request);request['_actual_request_sha256']=sys.argv[2]
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',IPFS_ACCEL_SKIP_CORE='1',
        IPFS_AUTO_INSTALL='false',IPFS_DATASETS_AUTO_INSTALL='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(D))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    need(Path(sys.modules[ParentedPhaseResourceReservation.__module__].__file__).absolute()
         == D/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py'
         and Path(open_existing_scheduler.__code__.co_filename).absolute()
         == D/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/existing_resource_scheduler.py'
         and Path(sys.modules[ResourceLane.__module__].__file__).absolute()
         == D/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py',
         'actual_held_native_sdk_modules_required')
    cancellation=threading.Event()
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(number,lambda _n,_f:cancellation.set())
    scheduler=open_existing_scheduler(STATE,POLICY)
    need(not os.path.lexists(RUN) and RUN.parent.is_dir() and not any(p.is_symlink() for p in (RUN.parent,*RUN.parent.parents)),
         'fresh_absent_genuine_fixture_root_required')
    with scheduler.acquire(ResourceLane.ORCHESTRATION,cpu_slots=1,memory_mb=1024,child_process_slots=8,
            timeout=3600,cancel_event=cancellation,request_id='legal-ir-freezer-git-component-r01') as root:
        with root.acquire_child(lane=ResourceLane.ORCHESTRATION,cpu_slots=1,memory_mb=1024,child_process_slots=8,
                timeout=3600,cancel_event=cancellation,request_id='legal-ir-freezer-git-component-r01:phase') as phase:
            RUN.mkdir(mode=0o700);attempt=RUN/'component';attempt.mkdir(mode=0o700);fsync_dir(RUN);fsync_dir(RUN.parent)
            with ParentedPhaseResourceReservation(LEDGER,roots=ROOTS,attempt_directory=attempt,parent_lease=phase,
                    storage_bytes=512*1024*1024,memory_mb=512,cpu_slots=1,child_process_slots=4,
                    scheduler_state_path=STATE,expected_scheduler_policy_sha256=POLICY,timeout_seconds=3600,
                    ledger_lock_timeout_seconds=None) as owner:
                guard(request)
                try:result=component(owner,request)
                except BaseException as error:
                    result={'schema':'legal-ir-report-freezer-git-component@1','passed':False,'scope':'synthetic Git component only',
                        'failure_type':type(error).__name__,'failure_reason':str(error)[:2048],
                        'source_request_sha256':sys.argv[2],'freezer_source_manifest_sha256':FREEZER_SHA,**FALSE}
                guard(request)
                write_new(attempt/'git-component-result.json',wire(result));fsync_tree(owner)
                finalized=owner.finalize(attempt,artifacts_durable=True)
                need(finalized.get('status')=='released','actual_native_component_owner_not_finalized')
                final={'reservation_id':finalized['reservation_id'],'status':finalized['status'],
                    'native_lease_id':finalized['resource_lease']['lease_id'],'native_lease_released':finalized['resource_lease']['released']}
    result['actual_owner_finalization']=final
    result['actual_root_closeout']={'lease_id':root.lease_id,'released':root.released}
    result['actual_phase_closeout']={'lease_id':phase.lease_id,'released':phase.released}
    need(root.released and phase.released,'actual_root_phase_release_not_observed')
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
