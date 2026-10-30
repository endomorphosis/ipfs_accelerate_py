"""Immutable report-only denial journals; JSON never grants productive authority.

The low-level store provides private-filesystem durability and read-only denial.
The original-object join is a disabled source proposal. It never invokes native
owners, providers, lifecycle quarantine publication, cleanup or borrowed exit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import threading
from types import ModuleType

SCHEMA = 'report-only-attempt-disposition-denial@1'
JOURNAL_DIRECTORY = 'report-only-attempt-dispositions'
QUARANTINE_REASON = 'report_only_attempt_disposition_pending'
BOARD_SHA256 = '5244288f69562663f44b00d27add1ee6ef338f977fed1c0e1c8855725ff372c7'
DAEMON_MODULE = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'
LIFECYCLE_MODULE = 'ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle'
MAX_DOCUMENT = 65536
MAX_RECORDS = 256
MAX_STORE_ENTRIES = 4096
MAX_SOURCE = 8*1024*1024
SHA = re.compile(r'[0-9a-f]{64}\Z')
NAME = re.compile(r'[0-9a-f]{64}\.json\Z')
TEMP = re.compile(r'\.pending-[0-9a-f]{32}\Z')
SCOPE_FIELDS = {'repo_root','state_path','task_id','canonical_task_cid','board_namespace','board_sha256'}
DENIAL_FIELDS = {'schema','scope','attempt','lifecycle','kind','reason_code','source_sha256',
                 'retry_authorized','settlement_authority','completion_authority','cleanup_allowed',
                 'acceptance_authority','parking_id'}
FALSE_FIELDS = ('retry_authorized','settlement_authority','completion_authority','cleanup_allowed','acceptance_authority')


class AttemptDispositionRefusal(RuntimeError):
    pass


class AttemptDispositionDenied(AttemptDispositionRefusal):
    def __init__(self, parking_ids):
        super().__init__('original_report_only_attempt_is_parked')
        self.parking_ids = tuple(parking_ids)
        self.retry_authorized = self.acceptance_authority = False


def require(value, reason):
    if not value:
        raise AttemptDispositionRefusal(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def token(value, *, bound=1024):
    require(type(value) is str and 0 < len(value.encode('utf-8')) <= bound
            and value == value.strip() and all(ord(c) >= 33 and ord(c) != 127 for c in value),
            'bounded_nonempty_identity_token_required')
    return value


def path_spelling(value):
    require(type(value) is str and len(value.encode()) <= 4096,'bounded_absolute_path_required')
    path = Path(value)
    require(path.is_absolute() and str(path) == value == os.path.abspath(value)
            and '..' not in path.parts,'canonical_absolute_path_spelling_required')
    return path


def canonical(value):
    path = path_spelling(str(value))
    require(not any(item.is_symlink() for item in (path,*path.parents)), 'unaliased_original_path_required')
    return path


def closed(value, depth=0):
    require(depth <= 8,'bounded_closed_JSON_depth_required')
    if value is None or type(value) in {bool,int}:
        require(type(value) is not int or abs(value) <= 2**63-1,'bounded_JSON_integer_required')
        return
    if type(value) is str:
        require(len(value.encode('utf-8')) <= 8192,'bounded_JSON_string_required')
        return
    if type(value) is list:
        require(len(value) <= MAX_RECORDS,'bounded_JSON_list_required')
        for item in value:closed(item,depth+1)
        return
    require(type(value) is dict and len(value) <= 64,'closed_JSON_object_required')
    for key,item in value.items():
        require(type(key) is str and len(key.encode()) <= 4096,'bounded_JSON_key_required')
        closed(item,depth+1)


def wire(value):
    closed(value)
    raw = json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()+b'\n'
    require(len(raw) <= MAX_DOCUMENT,'bounded_denial_document_required')
    return raw


def strict(raw):
    def pairs(rows):
        result = {}
        for key,value in rows:
            require(key not in result,'duplicate_denial_JSON_key');result[key] = value
        return result
    def refuse(_):
        raise AttemptDispositionRefusal('noninteger_or_nonfinite_denial_JSON_refused')
    try:
        value = json.loads(raw,object_pairs_hook=pairs,parse_float=refuse,parse_constant=refuse)
    except (ValueError,UnicodeError) as error:
        raise AttemptDispositionRefusal('invalid_denial_JSON') from error
    closed(value)
    return value


def validate_scope(scope):
    require(type(scope) is dict and set(scope) == SCOPE_FIELDS,'closed_task_revision_scope_required')
    for field in ('repo_root','state_path'):path_spelling(scope[field])
    for field in ('task_id','canonical_task_cid','board_namespace'):token(scope[field])
    require(type(scope['board_sha256']) is str and SHA.fullmatch(scope['board_sha256']), 'exact_board_digest_required')
    return scope


def revision_key(scope):
    """Canonical revision key excludes ordinal, display alias and board status."""
    validate_scope(scope)
    identity = {name:scope[name] for name in ('repo_root','state_path','canonical_task_cid','board_namespace')}
    return sha(wire(identity))


def validate_denial(value):
    closed(value)
    require(type(value) is dict and set(value) == DENIAL_FIELDS and value['schema'] == SCHEMA,
            'closed_denial_schema_required')
    validate_scope(value['scope'])
    require(type(value['kind']) is str and value['kind'] in {'parked_failure','candidate_pending_validation'},
            'inert_disposition_kind_required')
    require(type(value['reason_code']) is str and re.fullmatch('[a-z][a-z0-9_]{0,127}',value['reason_code']),
            'bounded_disposition_reason_required')
    require(all(value[name] is False for name in FALSE_FIELDS),'denial_must_grant_no_authority')
    attempt = value['attempt']
    require(type(attempt) is dict and set(attempt) == {'ordinal','task_count','cid_count','configured_cap'}
            and all(type(number) is int and 0 < number <= 1000000 for number in attempt.values())
            and attempt['ordinal'] == attempt['task_count'] == attempt['cid_count']
            and attempt['ordinal'] <= attempt['configured_cap'],'exact_consumed_original_ordinal_required')
    lifecycle = value['lifecycle']
    require(type(lifecycle) is dict and set(lifecycle) == {'record_id','lease_id','fence','workspace_path',
                'branch','owner_pid','owner_birth','owner_boot_id','snapshot_sha256'},'closed_original_lifecycle_binding_required')
    for field in ('record_id','lease_id','branch','owner_boot_id'):token(lifecycle[field])
    path_spelling(lifecycle['workspace_path'])
    require(all(type(lifecycle[field]) is int and 0 < lifecycle[field] <= 2**63-1
                for field in ('fence','owner_pid','owner_birth')),'positive_original_lifecycle_identity_required')
    require(type(lifecycle['snapshot_sha256']) is str and SHA.fullmatch(lifecycle['snapshot_sha256']),
            'exact_original_lifecycle_snapshot_digest_required')
    sources = value['source_sha256']
    require(type(sources) is dict and 1 <= len(sources) <= 16,'bounded_source_closure_required')
    for name,digest in sources.items():
        path_spelling(name)
        require(type(digest) is str and SHA.fullmatch(digest),'exact_original_source_digest_required')
    require(type(value['parking_id']) is str and SHA.fullmatch(value['parking_id']), 'exact_parking_identity_required')
    body = {name:item for name,item in value.items() if name != 'parking_id'}
    require(value['parking_id'] == sha(wire(body)),'denial_content_identity_changed')
    return value


def make_denial(*,scope,attempt,lifecycle,kind,reason_code,source_sha256):
    """Construct deny metadata only. It cannot assert original-object authority."""
    body = dict(schema=SCHEMA,scope=scope,attempt=attempt,lifecycle=lifecycle,kind=kind,
                reason_code=reason_code,source_sha256=source_sha256,
                **{name:False for name in FALSE_FIELDS})
    body = strict(wire(body))
    body['parking_id'] = sha(wire(body))
    return validate_denial(body)


def _directory_witness(info):
    return info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_gid


class DispositionJournal:
    """Private local journal, immutable compare-and-publish and deny-only reads."""
    def __init__(self,directory):
        self.directory = canonical(directory)
        self.fd = os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            self.identity = _directory_witness(os.fstat(self.fd))
            self._directory()
        except BaseException:
            os.close(self.fd);self.fd = None;raise

    def _directory(self):
        require(self.fd is not None,'journal_is_closed')
        canonical(self.directory)
        observed = os.fstat(self.fd)
        require(stat.S_ISDIR(observed.st_mode) and stat.S_IMODE(observed.st_mode) == 0o700
                and observed.st_uid == os.geteuid()
                and _directory_witness(observed) == self.identity
                == _directory_witness(self.directory.lstat()),'original_private_journal_directory_changed')

    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd = None

    def __enter__(self):return self

    def __exit__(self,*_):self.close()

    def __reduce__(self):raise AttemptDispositionRefusal('journal_descriptor_not_serializable')

    def _names(self,*,allow_owned_temporary=None):
        self._directory()
        names = []
        with os.scandir(self.fd) as entries:
            for entry in entries:
                require(len(names) < MAX_RECORDS,'bounded_journal_inventory_required')
                if allow_owned_temporary is not None and entry.name == allow_owned_temporary:
                    continue
                require(not TEMP.fullmatch(entry.name),'incomplete_journal_publication_retained')
                require(NAME.fullmatch(entry.name),'unknown_journal_entry_refused')
                names.append(entry.name)
        self._directory()
        return tuple(sorted(names))

    def _read(self,name):
        require(NAME.fullmatch(name),'exact_revision_journal_name_required')
        self._directory()
        try:
            fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=self.fd)
        except FileNotFoundError:return None
        with os.fdopen(fd,'rb') as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o600
                    and before.st_uid == os.geteuid() and before.st_nlink == 1
                    and 0 < before.st_size <= MAX_DOCUMENT,'strict_owned_single_link_journal_required')
            raw = stream.read(MAX_DOCUMENT+1);after = os.fstat(stream.fileno())
        witness = lambda row: (row.st_dev,row.st_ino,row.st_mode,row.st_uid,row.st_gid,
                               row.st_nlink,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        require(len(raw) == before.st_size and witness(before) == witness(after)
                == witness(os.stat(name,dir_fd=self.fd,follow_symlinks=False)), 'journal_replaced_during_read')
        value = validate_denial(strict(raw))
        require(raw == wire(value) and name == revision_key(value['scope'])+'.json',
                'canonical_revision_journal_bytes_required')
        self._directory()
        return value

    def publish(self,denial):
        """No overwrite; same exact bytes are idempotent, conflicts remain intact."""
        denial = validate_denial(strict(wire(denial)))
        raw = wire(denial);name = revision_key(denial['scope'])+'.json'
        names = self._names()
        prior = self._read(name)
        if prior is not None:
            require(wire(prior) == raw,'existing_revision_denial_conflicts')
            os.fsync(self.fd)
            return prior
        require(len(names) < MAX_RECORDS,'bounded_journal_inventory_required')
        temporary = '.pending-'+secrets.token_hex(16)
        fd = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self.fd)
        owned = os.fstat(fd)
        try:
            with os.fdopen(fd,'wb') as stream:
                os.fchmod(stream.fileno(),0o600)
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            self._directory()
            self._names(allow_owned_temporary=temporary)
            try:
                os.link(temporary,name,src_dir_fd=self.fd,dst_dir_fd=self.fd,follow_symlinks=False)
            except FileExistsError:
                pass  # A concurrent publisher's original bytes are never replaced.
        finally:
            named = os.stat(temporary,dir_fd=self.fd,follow_symlinks=False)
            require((named.st_dev,named.st_ino) == (owned.st_dev,owned.st_ino),
                    'original_journal_temporary_identity_changed')
            os.unlink(temporary,dir_fd=self.fd)
            os.fsync(self.fd)
        published = self._read(name)
        require(published is not None and wire(published) == raw,'published_revision_denial_conflicts_or_missing')
        return published

    def denial_for_revision(self,scope,*,prospective_attempt=None):
        validate_scope(scope)
        if prospective_attempt is not None:
            require(type(prospective_attempt) is int and 0 < prospective_attempt <= 1000000,
                    'bounded_prospective_attempt_required')
        names = self._names()
        name = revision_key(scope)+'.json'
        value = self._read(name)
        require(value is not None or name not in names,'journal_revision_disappeared_during_read')
        require(value is None or value['scope'] == scope,'journal_revision_scope_changed')
        return value  # No ordinal filtering: later attempts remain denied.

    def denials_for_state(self,state_path):
        state_path = str(path_spelling(str(state_path)))
        names = self._names()
        rows = []
        for name in names:
            value = self._read(name)
            require(value is not None,'journal_inventory_changed_during_read')
            if value['scope']['state_path'] == state_path:rows.append(value)
        require(self._names() == names,'journal_inventory_changed_during_read')
        return tuple(rows)

    def assert_no_denials_for_state(self,state_path):
        rows = self.denials_for_state(state_path)
        if rows:raise AttemptDispositionDenied(row['parking_id'] for row in rows)
        return None  # Absence of a deny is never a productive grant.


def _read_source(path,expected):
    path = canonical(path)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_uid in {0,os.geteuid()} and not before.st_mode & 2
                and 0 < before.st_size <= MAX_SOURCE,'strict_bounded_original_source_required')
        raw = stream.read(MAX_SOURCE+1);after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev,row.st_ino,row.st_size,row.st_mode,row.st_uid,row.st_gid,
                           row.st_nlink,row.st_mtime_ns,row.st_ctime_ns)
    require(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat())
            and sha(raw) == expected,'original_source_pin_changed')
    return raw


def _bounded_namespace(directory,*,maximum=MAX_STORE_ENTRIES):
    """Read-only preflight before original strict receipt inventory allocates."""
    require(type(maximum) is int and 0 < maximum <= MAX_STORE_ENTRIES,'finite_original_store_entry_bound_required')
    path = canonical(directory)
    try:fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    except FileNotFoundError:return None
    try:
        before = os.fstat(fd)
        require(stat.S_ISDIR(before.st_mode) and before.st_uid == os.geteuid()
                and not before.st_mode & 0o022,'owned_original_lifecycle_directory_required')
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                require(len(names) < maximum,'original_store_namespace_bound_exceeded')
                names.append(entry.name)
        after = os.fstat(fd)
        witness = lambda row: (row.st_dev,row.st_ino,row.st_mode,row.st_uid,row.st_gid,
                               row.st_mtime_ns,row.st_ctime_ns)
        canonical(path)
        require(witness(before) == witness(after) == witness(path.lstat()),
                'original_lifecycle_namespace_changed_during_preflight')
        return witness(after),tuple(sorted(names))
    finally:os.close(fd)


class OriginalDispositionScope:
    """Actual original daemon/store association; strict read-only lane deny gate."""
    def __init__(self,daemon_module,lifecycle_module,daemon,*,source_sha256):
        require(type(daemon_module) is ModuleType and daemon_module.__name__ == DAEMON_MODULE
                and sys.modules.get(DAEMON_MODULE) is daemon_module
                and type(lifecycle_module) is ModuleType and lifecycle_module.__name__ == LIFECYCLE_MODULE
                and sys.modules.get(LIFECYCLE_MODULE) is lifecycle_module,'actual_loaded_original_modules_required')
        require(type(daemon) is daemon_module.PortalImplementationDaemon
                and type(daemon.worktree_lifecycle) is lifecycle_module.WorktreeLifecycleStore,
                'actual_original_daemon_and_lifecycle_store_required')
        self.dm,self.lm,self.daemon = daemon_module,lifecycle_module,daemon
        self.store = daemon.worktree_lifecycle
        self.pid,self.thread = os.getpid(),threading.get_ident()
        self.pins = dict(source_sha256)
        require(3 <= len(self.pins) <= 16 and all(str(canonical(path)) == path
                and type(digest) is str and SHA.fullmatch(digest) for path,digest in self.pins.items()),
                'exact_bounded_original_source_closure_required')
        for path in (daemon_module.__file__,lifecycle_module.__file__,__file__):
            require(str(Path(path).absolute()) in self.pins,'original_join_source_pin_required')
        self.state_path = canonical(daemon.state_path)
        self.directory = self.state_path.parent/JOURNAL_DIRECTORY
        self._sources()

    def __reduce__(self):raise AttemptDispositionRefusal('actual_original_disposition_scope_not_serializable')

    def _sources(self):
        require(self.pid == os.getpid() and self.thread == threading.get_ident()
                and sys.modules.get(DAEMON_MODULE) is self.dm and sys.modules.get(LIFECYCLE_MODULE) is self.lm
                and type(self.daemon) is self.dm.PortalImplementationDaemon
                and self.daemon.worktree_lifecycle is self.store
                and type(self.store) is self.lm.WorktreeLifecycleStore
                and canonical(self.daemon.state_path) == self.state_path
                and canonical(self.store.repo_root) == canonical(self.daemon.repo_root),
                'actual_original_scope_identity_changed')
        for name,digest in self.pins.items():_read_source(name,digest)

    def revision_for_task(self,task):
        self._sources()
        require(type(task) is self.dm.PortalTask,'actual_original_PortalTask_required')
        identity = self.daemon._identity_for_task(task)
        board = canonical(self.daemon.todo_path)
        _read_source(board,BOARD_SHA256)
        return validate_scope(dict(repo_root=str(canonical(self.daemon.repo_root)),state_path=str(self.state_path),
            task_id=task.task_id,canonical_task_cid=identity.canonical_task_cid,
            board_namespace=identity.board_namespace,board_sha256=BOARD_SHA256))

    def assert_no_denials_for_state(self):
        self._sources()
        parent = self.state_path.parent.lstat()
        require(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid()
                and stat.S_IMODE(parent.st_mode) == 0o700,'original_private_state_directory_required')
        try:self.directory.lstat()
        except FileNotFoundError:pass
        else:
            with DispositionJournal(self.directory) as journal:
                journal.assert_no_denials_for_state(self.state_path)
        # A missing journal is not absence of an original immutable parked deny.
        parking_ids = []
        namespace = _bounded_namespace(self.store.store_dir)
        quarantines = self.store.iter_quarantines()
        require(_bounded_namespace(self.store.store_dir) == namespace,
                'original_strict_quarantine_namespace_changed')
        require(type(quarantines) is tuple and len(quarantines) <= MAX_RECORDS,
                'bounded_original_strict_quarantine_inventory_required')
        for receipt in quarantines:
            authority = receipt['fence_authority']
            if authority.get('schema') != SCHEMA:continue
            validate_denial(authority)
            require(receipt['lifecycle_record']['record_id'] == authority['lifecycle']['record_id'],
                    'original_quarantine_denial_lifecycle_changed')
            if authority['scope']['state_path'] == str(self.state_path):parking_ids.append(authority['parking_id'])
        if parking_ids:raise AttemptDispositionDenied(parking_ids)
        self._sources()
        return None

    def bind_attempt(self,task,state,captured):
        return OriginalAttemptDisposition(self,task,state,captured)


class OriginalAttemptDisposition:
    """Original objects retained; journal publish and quarantine join deny only."""
    def __init__(self,scope,task,state,captured):
        require(type(scope) is OriginalDispositionScope,'actual_original_scope_required')
        self.scope,self.task,self.state,self.captured = scope,task,state,captured
        self.denial_bytes = None
        self.revision = scope.revision_for_task(task)
        self._original()

    def __reduce__(self):raise AttemptDispositionRefusal('actual_original_attempt_binding_not_serializable')

    def _original(self):
        scope = self.scope;scope._sources()
        daemon,captured,state = scope.daemon,self.captured,self.state
        require(type(state) is scope.dm.PortalTaskState and type(captured) is scope.lm.WorkspaceLifecycleRecord
                and daemon._active_worktree_lifecycle is captured and not captured.is_terminal
                and daemon._database_attempt_authority is None
                and daemon.report_only_candidate_freezing is True and daemon.preserve_task_attempts is True
                and daemon.use_ephemeral_worktree is True,'actual_original_report_only_attempt_required')
        require(scope.revision_for_task(self.task) == self.revision
                and captured.task_id == self.task.task_id
                and captured.canonical_task_cid == self.revision['canonical_task_cid']
                and state.active_task_id == self.task.task_id
                and state.active_task_cid == captured.canonical_task_cid
                and state.active_attempt == captured.attempt and state.implementation_in_progress is True
                and canonical(state.active_worktree_path) == canonical(captured.workspace_path)
                and state.active_branch == captured.branch
                and state.implementation_attempts.get(self.task.task_id) == captured.attempt
                and state.implementation_attempts_by_cid.get(captured.canonical_task_cid) == captured.attempt
                and type(daemon.max_task_attempts) is int and captured.attempt <= daemon.max_task_attempts,
                'exact_original_consumed_task_revision_and_state_required')
        require(captured.owner == scope.lm.current_process_birth(proc_root=scope.store.proc_root)
                and captured.owner.pid == scope.pid
                and str(canonical(captured.repo_root)) == self.revision['repo_root']
                and str(canonical(captured.state_dir)) == str(scope.state_path.parent)
                and canonical(daemon.worktree_root) in canonical(captured.workspace_path).parents
                and canonical(captured.workspace_path) != canonical(daemon.repo_root)
                and captured.record_id == captured.compute_record_id()
                and scope.store.load_workspace(captured.workspace_path) == captured,
                'actual_live_original_lifecycle_capture_required')
        return daemon,captured,state

    def publish_failure(self,*,reason_code):
        daemon,captured,state = self._original()
        record_bytes = (json.dumps(captured.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        denial = make_denial(scope=self.revision,
            attempt=dict(ordinal=captured.attempt,task_count=state.implementation_attempts[self.task.task_id],
                         cid_count=state.implementation_attempts_by_cid[captured.canonical_task_cid],
                         configured_cap=daemon.max_task_attempts),
            lifecycle=dict(record_id=captured.record_id,lease_id=captured.lease_id,fence=captured.fence,
                           workspace_path=str(canonical(captured.workspace_path)),branch=captured.branch,
                           owner_pid=captured.owner.pid,owner_birth=captured.owner.start_time_ticks,
                           owner_boot_id=captured.owner.boot_id,snapshot_sha256=sha(record_bytes)),
            kind='parked_failure',reason_code=reason_code,source_sha256=self.scope.pins)
        with DispositionJournal(self.scope.directory) as journal:published = journal.publish(denial)
        self.denial_bytes = wire(published)
        return strict(self.denial_bytes)

    def quarantine_authority(self):
        """Denial provenance for original API; caller still owns its admission."""
        self._original()
        require(self.denial_bytes is not None,'original_durable_failure_journal_required')
        with DispositionJournal(self.scope.directory) as journal:
            current = journal.denial_for_revision(self.revision)
        require(current is not None and wire(current) == self.denial_bytes,'original_published_failure_denial_changed')
        return strict(self.denial_bytes)

    def join_original_quarantine(self):
        """Read actual original receipt; never publish quarantine or close borrow."""
        authority = self.quarantine_authority()
        receipt = self.scope.store.load_quarantine(self.captured.workspace_path)
        require(type(receipt) is dict and receipt['lifecycle_record'] == self.captured.to_dict()
                and receipt['fence_authority'] == authority
                and receipt['reason'] == QUARANTINE_REASON
                and all(receipt[name] is False for name in ('cleanup_allowed','reuse_allowed',
                       'evidence_reuse_allowed','terminalized','worktree_deleted','branch_deleted')),
                'matching_original_immutable_quarantine_required')
        return dict(parking_id=authority['parking_id'],original_quarantine_id=receipt['quarantine_id'],
            durable_failure_denial_observed=True,original_borrow_retained=True,
            physical_custody_authority=False,cleanup_allowed=False,acceptance_authority=False,
            completion_authority=False,canonical_tasks_accepted=0)
