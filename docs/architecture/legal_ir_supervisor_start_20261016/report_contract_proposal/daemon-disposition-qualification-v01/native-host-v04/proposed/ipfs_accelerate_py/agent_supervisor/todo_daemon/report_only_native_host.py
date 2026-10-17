"""Opt-in same-PID host proposal. No new root, state connection or CLI grant."""
from contextvars import ContextVar
import hashlib
import importlib
import os
from pathlib import Path
import stat
import sys
import threading
import time

NATIVE = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
RUNTIME = 'ipfs_accelerate_py.agent_supervisor.runtime.'
TODO = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.'
DATASETS = Path('/home/barberb/lift_coding/external/ipfs_datasets')
ROOTS = (DATASETS/'workspace/test-logs', DATASETS/'workspace/todo-queues', DATASETS/'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
LEDGER = ROOTS[0]/'federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
DAEMON_SHA = '4984370456bd304dd9d89550c05e2d364fd9f0ca4849db560cb10caddfe3f407'
MIB = 1024*1024
_HOST = ContextVar('actual_report_only_native_host', default=None)
_RETAINED = {}


class NativeHostRefusal(RuntimeError):
    pass


class NativeHostCustodyPending(BaseException):
    """Retain original native/connection/task/claim frames; never serialize."""
    def __init__(self, host, cause):
        super().__init__('actual_report_only_host_custody_pending')
        self.actual_host, self.actual_cause = host, cause
    def __reduce__(self):
        raise NativeHostRefusal('actual_host_custody_not_serializable')


def need(value, reason):
    if not value:
        raise NativeHostRefusal(reason)


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased_absolute_host_path_required')
    return path


def require_report_only_native_host(daemon):
    host = _HOST.get()
    need(type(host) is ReportOnlyNativeHost and getattr(daemon, '_report_only_native_host', None) is host,
         'explicit_same_process_native_host_required')
    # The public daemon resolves its host before guard_pass_entry. Keep this
    # accessor effect-free so sticky refusal precedes borrowed DB witnesses.
    host._binding_identity(daemon)
    return host


def _success_callback(outcome):
    host = _HOST.get()
    need(type(host) is ReportOnlyNativeHost, 'original_active_host_callback_required')
    host._freeze(outcome)


class ReportOnlyNativeHost:
    def __init__(self, daemon, *, parent_phase, control_plane_connection, control_plane_target,
                 attempt_root, protected_source_sha256, custody_source_sha256):
        need(sys.dont_write_bytecode and threading.current_thread() is threading.main_thread(),
             'single_main_thread_no_bytecode_host_required')
        self.daemon, self.phase, self.connection, self.target = daemon, parent_phase, control_plane_connection, control_plane_target
        self.pins, self.custody_pins = dict(protected_source_sha256), dict(custody_source_sha256)
        need(0 < len(self.custody_pins) <= 32 and all(self.pins.get(path) == digest for path,digest in self.custody_pins.items()),
             'exact_custody_source_subset_required')
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.root = canonical(attempt_root)
        need(any(root in self.root.parents for root in ROOTS) and not os.path.lexists(self.root), 'fresh_named_root_host_attempt_required')
        self.owners, self.consumers, self.files, self.frames, self.role_owners = [], [], [], [], {}
        self.borrow, self.token, self.hold, self.pending = None, None, None, None
        self.git_pending, self.ordinary_pending = None, None
        self.task, self.state, self.command, self.args, self.workspace, self.invocation = (None,)*6
        self.result, self.bound, self.finished = None, False, False
        self.disposition_pending = False
        self._execution_attempted = False
        self._pass_entry_attempted = False
        self._pass_entry_validated = False
        self.finalizers = None
        self.storage_roots = None
        self.dispatch_attempted = False
        self.handoff_module, self.freezer_module = None, None
        self._sources()
        self._phase_live()
        module = self._module(TODO+'implementation_daemon')
        need(type(daemon) is module.PortalImplementationDaemon and self.pins[str(Path(module.__file__).absolute())] == DAEMON_SHA,
             'actual_composed_original_daemon_required')
        continuations = self._module(TODO+'original_finalizer_continuations')
        self.finalizers = continuations.OriginalFinalizerContinuations(module, daemon)

    def __reduce__(self):
        raise NativeHostRefusal('actual_host_and_native_references_not_serializable')

    def __copy__(self):
        raise NativeHostRefusal('actual_host_not_copyable')

    def __deepcopy__(self, memo):
        raise NativeHostRefusal('actual_host_not_copyable')

    def _sources(self):
        need(0 < len(self.pins) <= 128 and str(Path(__file__).absolute()) in self.pins, 'complete_bounded_host_source_pins_required')
        for name, digest in self.pins.items():
            path = canonical(name)
            need(type(digest) is str and len(digest) == 64, 'exact_host_source_digest_required')
            fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                before = os.fstat(stream.fileno())
                need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
                     and before.st_mode & 2 == 0 and 0 < before.st_size <= 256*MIB, 'bounded_owned_host_source_required')
                digestor = hashlib.sha256(); count = 0
                while chunk := stream.read(MIB):
                    count += len(chunk); need(count <= before.st_size, 'host_source_grew'); digestor.update(chunk)
                after = os.fstat(stream.fileno())
            witness = lambda row: (row.st_dev,row.st_ino,row.st_mode,row.st_uid,row.st_gid,row.st_nlink,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
            need(count == before.st_size and witness(before) == witness(after) == witness(path.lstat())
                 and digestor.hexdigest() == digest, 'host_source_changed')

    def _module(self, name):
        suffix = '/'+name.replace('.', '/')+'.py'
        paths = [path for path in self.pins if path.endswith(suffix)]
        need(len(paths) == 1, 'exact_host_module_pin_required')
        self._sources()
        module = importlib.import_module(name)
        need(str(Path(module.__file__).absolute()) == paths[0], 'actual_loaded_host_module_differs')
        if name == RUNTIME+'ordinary_success_handoff':self.handoff_module = module
        if name == TODO+'report_only_candidate_freezer':self.freezer_module = module
        return module

    def _phase_live(self):
        resource = self._module(NATIVE+'resource_scheduler')
        need(type(self.phase) is resource.ResourceLease and self.phase.owner_pid == self.pid
             and self.phase.parent_lease_id is not None and self.phase.lane == 'orchestration'
             and self.phase.cpu_slots == 3 and self.phase.memory_mb == 4096 and self.phase.child_process_slots == 16
             and Path(self.phase.token.state_path) == STATE and self.phase.renew() and not self.phase.cancelled
             and not self.phase.released, 'actual_already_admitted_same_PID_three_CPU_sixteen_slot_phase_required')
        existing = self._module(NATIVE+'existing_resource_scheduler')
        rows = {row['lease_id']: row for row in existing.open_existing_scheduler(STATE,POLICY).active_leases()}
        root = rows.get(self.phase.parent_lease_id)
        need(root is not None and root.get('parent_lease_id') is None and root.get('owner_pid') == self.pid
             and root.get('cpu_slots') == 3 and root.get('memory_mb') == 4096 and root.get('child_process_slots') == 16,
             'original_unchanged_root_phase_profile_required')

    def _binding_identity(self, daemon):
        # Pure actual-object/process identity only; no source import or SQL.
        need(daemon is self.daemon and self.pid == os.getpid() and self.thread == threading.get_ident()
             and self.bound and not self.finished, 'actual_same_PID_thread_host_scope_required')

    def _identity(self, daemon):
        self._binding_identity(daemon)
        try:
            state = self._module(TODO+'portal_task_state_control_plane')
            need(state.require_borrowed_task_state_connection(daemon.state_path, connection=self.connection) is self.connection,
                 'original_borrowed_connection_identity_required')
        except Exception as error:
            if self.frames or self.owners or self.pending is not None:self._retain(error)
            raise

    def execute_daemon_once(self):
        need(self.pending is None and self.disposition_pending is False,
             'retained_host_cannot_start_productive_execution')
        need(self._execution_attempted is False and not self.bound and not self.finished
             and self.token is None and _HOST.get() is None, 'fresh_single_host_pass_required')
        self._execution_attempted = True
        self._phase_live()
        state = self._module(TODO+'portal_task_state_control_plane')
        self.borrow = state.borrow_task_state_connection(self.daemon.state_path, connection=self.connection, target=self.target)
        self.borrow.__enter__()
        self.bound = True; self.token = _HOST.set(self); self.daemon._report_only_native_host = self
        try:
            result = self.daemon.run_once()
            self.before_original_finalizer(self.daemon,stage='host_exit',original_frame={})
        except BaseException as error:
            self.frames.append(error)
            if isinstance(error, NativeHostCustodyPending) or self.pending is not None:
                # Stopped resources do not resume the original unwound claim/fence finalizers.
                self._retain(error)
            if not self.observe_pending():
                self._retain(error)
            self._finish_borrow()
            raise
        self._finish_borrow()
        return result

    def _finish_borrow(self):
        try:
            need(self.pending is None, "original_finalizer_settlement_required_before_borrow_exit")
            need(not self.disposition_pending, 'original_attempt_disposition_required_before_borrow_exit')
            need(self.finalizers is not None and self.finalizers.complete(), 'all_original_finalizers_required_before_borrow_exit')
            need(self.observe_pending(), 'actual_host_must_be_stopped_before_borrow_exit')
            self.borrow.__exit__(None,None,None)  # Borrow unregisters; never closes caller connection.
        except BaseException as error:
            self.frames.append(error)
            self._retain(error)
        self.daemon._report_only_native_host = None
        _HOST.reset(self.token); self.bound = False; self.finished = True
        _RETAINED.pop(id(self),None)

    def guard_pass_entry(self, daemon):
        # execute_daemon_once's fresh-binding check is not enough: a retained
        # pending host remains bound, so public daemon.run_once can resolve it
        # again without entering execute_daemon_once. Never renew that pass.
        need(self.pending is None and self.disposition_pending is False,
             'retained_host_cannot_enter_productive_pass')
        need(self._pass_entry_attempted is False and self._pass_entry_validated is False,
             'original_single_public_pass_already_attempted')
        # Latch BEFORE any fallible source, borrowed DB or phase checks. A
        # refused/failed first check cannot become another productive pass.
        self._pass_entry_attempted = True
        need(self._execution_attempted is True, 'original_execute_scope_required_before_public_pass')
        self._validate_in_pass_profile(daemon)
        self._pass_entry_validated = True

    def _validate_in_pass_profile(self, daemon):
        # Repeated checks inside the already admitted pass must not consume or
        # reset its one-pass latch. Pending custody always stops productivity.
        need(self.pending is None and self.disposition_pending is False,
             'retained_host_cannot_validate_productive_profile')
        self._identity(daemon); self._phase_live()
        need(daemon.report_only_candidate_freezing is True and daemon.max_task_attempts == 3
             and type(daemon.max_task_attempts) is int and daemon.preserve_task_attempts is True
             and daemon.use_ephemeral_worktree is True and daemon.production_provider_policy is None
             and daemon.manual_completion_authority_revalidation_only is False,
             'explicit_unchanged_report_only_profile_required_before_original_pass')
        state = self._module(TODO+'portal_task_state_control_plane')
        state.require_borrowed_task_state_source_profile(daemon.state_path, task_source=daemon.task_source,
            plan_revision_store=daemon.plan_revision_store, require_active_plan_revision=daemon.require_active_plan_revision,
            configured_parallel_execution_plan=daemon._configured_parallel_execution_plan, active_plan_binding=daemon._active_plan_binding)

    def _require_validated_in_pass(self, daemon):
        # Necessary original in-memory conditions, never a portable pass grant.
        # Check denial/latches before borrowed DB witnesses or native renewal.
        need(self.pending is None and self.disposition_pending is False,
             'retained_host_cannot_perform_productive_in_pass_effect')
        need(self._execution_attempted is True and self._pass_entry_attempted is True
             and self._pass_entry_validated is True,
             'actual_original_pass_must_be_entered_and_validated')
        self._binding_identity(daemon)

    def require_attempt_profile(self, daemon, task, state):
        self._require_validated_in_pass(daemon)
        self._validate_in_pass_profile(daemon)
        module = self._module(TODO+'implementation_daemon')
        need(type(task) is module.PortalTask and type(state) is module.PortalTaskState
             and (self.task is None or self.task is task and self.state is state), 'original_single_task_state_objects_required')
        need(type(daemon.max_task_attempts) is int and daemon.max_task_attempts > 0 and daemon.preserve_task_attempts is True,
             'original_configured_attempt_cap_required')
        need(not daemon._manual_completion_authority_revalidation_only_task(task)
             and not daemon._task_uses_typed_local_execution(task)
             and daemon._retry_no_change_pre_dispatch_scope(task,state) is None,
             'ordinary_ephemeral_only_no_manual_deterministic_retry_probe_required')
        self.task, self.state = task, state

    def capture_attempt_frame(self, daemon, *, task, state, attempt, task_claim_path, task_claim_metadata,
                              acquired_resource_claims, lock_path, lock_metadata):
        self.require_attempt_profile(daemon,task,state)
        self.frames.append(dict(task=task,state=state,attempt=attempt,task_claim_path=task_claim_path,
            task_claim_metadata=task_claim_metadata,resource_claims=acquired_resource_claims,lock_path=lock_path,lock_metadata=lock_metadata))

    def _owner(self, role, memory, slots, storage):
        self._require_validated_in_pass(self.daemon)
        self._phase_live()
        self.root.mkdir(mode=0o700,exist_ok=True)
        path = self.root/role; path.mkdir(mode=0o700,exist_ok=False)
        native = self._module(NATIVE+'parented_phase_resources')
        owner = native.ParentedPhaseResourceReservation(LEDGER,roots=ROOTS,attempt_directory=path,parent_lease=self.phase,
            storage_bytes=storage*MIB,memory_mb=memory,cpu_slots=1,child_process_slots=slots,
            scheduler_state_path=STATE,expected_scheduler_policy_sha256=POLICY,timeout_seconds=3600,ledger_lock_timeout_seconds=None)
        self.owners.append(owner)  # Retain BEFORE __enter__ can allocate/fail.
        self.role_owners[role] = owner
        owner.__enter__()
        return owner

    def _storage(self, owner, roots):
        self._require_validated_in_pass(self.daemon)
        helper = self._module(TODO+'report_only_candidate_freezer')
        roots = tuple(canonical(path) for path in roots)
        need(0 < len(roots) <= 3 and len(set(roots)) == len(roots)
             and all(self.root not in path.parents and path != self.root and path not in self.root.parents for path in roots),
             'disjoint_original_external_storage_required')
        declarations = {str(path):128*MIB for path in roots}
        need(sum(declarations.values()) <= 384*MIB, 'bounded_freezer_external_charge_required')
        for path in roots: owner.account_external_bytes(helper.storage_charge_key(path),declarations[str(path)])
        return declarations

    def preflight_candidate(self, daemon, state, task, *, command):
        self.require_attempt_profile(daemon,task,state)
        need(self.command is None and type(command) is tuple, 'single_original_command_before_consumption_required')
        runner = self._module(RUNTIME+'grok_cli_runner')
        path = str(Path(runner.__file__).absolute())
        need(len(command) > 2 and canonical(command[0]) == Path(sys.executable).resolve()
             and (command[1] == path or tuple(command[1:3]) == ('-m',runner.__name__)), 'exact_packaged_primary_runner_command_required')
        argv = command[3:] if command[1] == '-m' else command[2:]
        try:args = runner.parse_runner_arguments(argv)
        except SystemExit as error:raise NativeHostRefusal('original_primary_runner_argv_refused_before_consumption') from error
        handoff = self._module(RUNTIME+'ordinary_success_handoff')
        handoff.check_primary_only_options(args,-1,environment=os.environ)
        selected_binary = str(getattr(args,'grok_bin','') or '').strip()
        need(selected_binary and Path(selected_binary).is_absolute()
             and str(getattr(args,'model','') or '').strip() in {'','grok-4.7'}
             and getattr(args,'mode','agent') == 'agent'
             and not str(getattr(args,'invocation_id','') or '').strip()
             and not str(getattr(args,'invocation_binding_sha256','') or '').strip(),
             'effect_free_primary_callable_profile_required_before_consumption')
        workspace = canonical(Path(args.workspace))
        trusted_binary = runner._resolve_trusted_grok_bin(configured=selected_binary,workspace=workspace)
        need(trusted_binary and trusted_binary in self.pins, 'actual_trusted_binary_pin_required_before_consumption')
        self._sources()  # Includes exact resolved binary; no CLI/version/auth probe.
        self.command, self.args = command, args
        owner = self._owner('preflight',512,4,512)
        try:
            self.storage_roots = tuple(daemon.report_only_preflight_storage_roots(state,task=task))
            declarations = self._storage(owner,self.storage_roots)
            daemon.preflight_report_only_candidate(state,task=task,preflight_owner=owner,parent_phase=self.phase,
                control_plane_connection=self.connection,protected_source_sha256=self.pins,declared_external_storage=declarations)
            self.workspace = canonical(daemon.report_only_preflight_workspace(state,task=task))
            need(Path(args.workspace).resolve(strict=True) == self.workspace, 'original_selected_command_workspace_differs')
        except BaseException as error:
            self.handle_attempt_exception(daemon,error,task=task,state=state)
            raise
        self._release(owner)

    def dispatch_provider(self, daemon, *, state, task, attempt, prompt, environment, log_path, command):
        self._require_validated_in_pass(daemon)
        self._identity(daemon)
        need(self.task is task and self.state is state and command == self.command and self.workspace is not None
             and self.invocation is None and not self.dispatch_attempted, 'original_preflight_command_and_single_dispatch_required')
        self.dispatch_attempted = True
        self.frames.append(dict(task=task,state=state,attempt=attempt,prompt=prompt,environment=environment,log_path=log_path,command=command))
        primary = self._owner('provider',1024,8,512); watchdog = self._owner('watchdog',512,4,128)
        temporary = self.root/'provider-temporary'; temporary.mkdir(mode=0o700)
        need(self.workspace != temporary and self.workspace not in temporary.parents and temporary not in self.workspace.parents,
             'disjoint_provider_workspace_and_private_temporary_root_required')
        declarations = [dict(path=str(path),bytes=128*MIB) for path in (self.workspace,temporary)]
        for row in declarations: primary.account_external_bytes(hashlib.sha256(row['path'].encode()).hexdigest(),row['bytes'])
        for owner in (primary,watchdog):
            self.consumers.append(owner.native_lease.acquire_child(lane='orchestration',cpu_slots=1,memory_mb=1,child_process_slots=0,timeout=30))
        custody = self._module(RUNTIME+'ordinary_custody_hold')
        self.hold = custody.bind_ordinary_custody_hold(primary_owner=primary,watchdog_owner=watchdog,
            primary_consumer=self.consumers[0],watchdog_consumer=self.consumers[1],protected_source_sha256=self.custody_pins)
        paths = (primary.directory/'stdout.bin',primary.directory/'stderr.bin')
        for path in paths:
            fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            self.files.append(os.fdopen(fd,'wb',buffering=0))
        runner = self._module(RUNTIME+'grok_cli_runner')
        self.invocation = runner.OrdinaryCallableInvocation(prompt,self.workspace,environment,temporary,
            paths[0],paths[1],self.files[0].fileno(),self.files[1].fileno(),2*MIB)
        daemon._mark_provider_launch_boundary(state,task=task,attempt=attempt,workspace_path=self.workspace,
            callable_invocation=self.invocation,control_plane_connection=self.connection)
        handoff = self._module(RUNTIME+'ordinary_success_handoff')
        try:
            code = handoff.run_ordinary_with_success_handoff(self.args,-1,owner=primary,watchdog_owner=watchdog,
                source_sha256=self.pins,declared_external_storage=declarations,on_success=_success_callback,
                invocation=self.invocation,custody_hold=self.hold,custody_source_sha256=self.custody_pins)
            need(code == 0 and self.result is not None, 'successful_actual_provider_and_candidate_required')
        except BaseException as error:
            return self.handle_attempt_exception(daemon,error,task=task,state=state)
        for file in self.files: file.flush(); os.fsync(file.fileno()); file.close()
        self._release(watchdog); self._release(primary)
        return self.result

    def _freeze(self, outcome):
        self._require_validated_in_pass(self.daemon)
        handoff = self._module(RUNTIME+'ordinary_success_handoff')
        capture = handoff.authenticate_ordinary_callback_for_freezer(outcome)  # Only12 owner slots, spare admin capacity.
        owner = None
        try:
            owner = self._owner('freezer',512,4,512)
            with handoff.authenticated_ordinary_freezer_scope(capture,freezer_owner=owner):
                try:
                    declaration = self._storage(owner,self.storage_roots)
                    candidate = self.daemon.freeze_report_only_candidate(outcome=outcome,freezer_owner=owner,
                        protected_source_sha256=self.pins,declared_external_storage=declaration)
                    self.result = dict(status='report_only_candidate_frozen',candidate=candidate,acceptance_authority=False,canonical_tasks_accepted=0)
                except BaseException as error:
                    self.frames.append(error)
                    helper = self.freezer_module
                    if helper is not None and type(error) is helper.GitCustodyPending:
                        self.git_pending = error  # Retain INSIDE authenticated scope before its finally.
                        try:
                            if helper.acknowledge_stopped_git_custody(error):self.git_pending = None
                        except Exception:pass
                    raise
                finally:
                    self._release(owner)  # SDK independently requires group0 and consumer drain.
        finally:
            if owner is None:
                handoff.discard_unused_authenticated_callback(capture)

    def _release(self, owner):
        public = owner.to_dict()
        if public['status'] == 'released':return
        if public['status'] == 'not_entered' and public.get('record') is None:
            lease = public.get('resource_lease')
            need(lease is None or lease.get('released') is True, 'unentered_native_allocation_outcome_unknown')
            return
        count = 0
        for directory,subdirs,files in os.walk(owner.directory,followlinks=False):
            count += len(subdirs)+len(files)
            need(count <= 16384, 'bounded_private_durability_required')
            for name in files:
                path = canonical(Path(directory)/name); meta = path.lstat()
                need(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1, 'regular_private_output_required')
                fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                try:os.fsync(fd)
                finally:os.close(fd)
            fd = os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
        owner.release(artifacts_durable=True)  # Same public stopped proof, including cancelled/expired ancestry.

    def _retain(self, cause):
        if self.pending is None:self.pending = NativeHostCustodyPending(self,cause)
        _RETAINED[id(self)] = self
        raise self.pending

    def observe_pending(self):
        try:
            need(self.pid == os.getpid() and self.thread == threading.get_ident(), 'original_host_observer_PID_thread_required')
            if not self.owners and not self.consumers and self.hold is None and self.git_pending is None and self.ordinary_pending is None:
                return True  # This exact source path has allocated nothing.
            helper = self._module(TODO+'report_only_candidate_freezer')
            if self.git_pending is not None:
                if not helper.acknowledge_stopped_git_custody(self.git_pending):return False
                self.git_pending = None
            # Freezer/preflight stops BEFORE ordinary observer may query Docker.
            for role in ('freezer','preflight'):
                if role in self.role_owners:self._release(self.role_owners[role])
            handoff = self._module(RUNTIME+'ordinary_success_handoff')
            if self.ordinary_pending is not None:
                if not handoff.acknowledge_stopped_ordinary_custody(self.ordinary_pending):return False
                self.ordinary_pending = None
            elif self.hold is not None and self.consumers and not all(row.released for row in self.consumers):
                handoff.acknowledge_unstarted_custody_hold(self.hold,primary_owner=self.role_owners['provider'],watchdog_owner=self.role_owners['watchdog'])
            elif self.hold is None:
                for role in ('provider','watchdog'):
                    if role not in self.role_owners:continue
                    owner = self.role_owners[role]
                    if owner.to_dict()['status'] == 'released':continue
                    record = owner.to_dict()['record']
                    need(record.get('child') is None and record.get('prior_children',[]) == [], 'actual_unbound_unstarted_native_history_required')
                for consumer in self.consumers:
                    if not consumer.released:consumer.release()
            for file in self.files:
                if not file.closed:file.flush();os.fsync(file.fileno());file.close()
            for owner in reversed(self.owners):self._release(owner)
            return True
        except Exception:return False

    def handle_attempt_exception(self, daemon, exc, *, task, state, lifecycle_record=None):
        self.frames.append(dict(exception=exc,task=task,state=state,lifecycle=lifecycle_record))
        if self.handoff_module is not None and type(exc) is self.handoff_module.OrdinaryCustodyPending:self.ordinary_pending = exc
        if self.freezer_module is not None and type(exc) is self.freezer_module.GitCustodyPending:self.git_pending = exc
        self._identity(daemon)
        if not self.observe_pending():self._retain(exc)
        self.result = dict(status='report_only_refused',failure_type=type(exc).__name__,acceptance_authority=False,canonical_tasks_accepted=0)
        return self.result

    def _latch_disposition(self):
        # This latch only retains more custody; it grants no transition/acceptance.
        self.disposition_pending = bool(self.disposition_pending
            or self.state is not None and self.state.implementation_in_progress
            or getattr(self.daemon, '_report_only_before_attempt', None) is not None
            or getattr(self.daemon, '_report_only_dispatch', None) is not None
            or getattr(self.daemon, '_active_worktree_lifecycle', None) is not None
            or self.result is not None and self.result.get('status') == 'report_only_candidate_frozen')

    def _drain_original_finalizers(self):
        try:
            return self.finalizers.drain()
        except BaseException as error:
            self.frames.append(error)
            self._retain(error)

    def _run_original_finalizer(self, daemon, *, stage, original_frame, continuation, caller_frame):
        # The public direct wrapper has already retained/authenticated the frame.
        self._identity(daemon)
        self._latch_disposition()
        if not self.observe_pending():
            self._retain(NativeHostRefusal('original_finalizer_requires_independent_stopped_closeout'))
        return self._drain_original_finalizers()

    def before_original_finalizer(self, daemon, *, stage, original_frame):
        # Only the old host-exit guard remains; original finally sites use exact continuations.
        self.frames.append((stage, original_frame))
        self._identity(daemon)
        need(stage == 'host_exit' and type(original_frame) is dict, 'explicit_host_exit_guard_only')
        self._latch_disposition()
        if not self.observe_pending():
            self._retain(NativeHostRefusal('original_finalizer_requires_independent_stopped_closeout'))
        self._drain_original_finalizers()
        if self.disposition_pending:
            self._retain(NativeHostRefusal('original_attempt_requires_explicit_parking_or_validation_handoff'))

    def settle_retained_finalizers(self):
        """Observe and run original stopped cleanup; never settle an attempt disposition.

        The caller must still hold the original actual root/phase and connection.
        No task IDs, replacement callbacks, result JSON or new native grant enter.
        """
        need(self.pending is not None and _RETAINED.get(id(self)) is self,
             'actual_original_retained_host_required')
        self._identity(self.daemon)
        self._latch_disposition()
        if not self.observe_pending():
            return dict(physical_stopped=False, original_finalizers_drained=False,
                original_borrow_retained=True, disposition_pending=self.disposition_pending,
                acceptance_authority=False, canonical_tasks_accepted=0)
        report = self._drain_original_finalizers()
        return dict(physical_stopped=True, original_finalizers_drained=self.finalizers.complete(),
            original_finalizers=report, original_borrow_retained=True,
            disposition_pending=self.disposition_pending, acceptance_authority=False,
            canonical_tasks_accepted=0)

    def require_candidate_result(self, daemon, result):
        self._identity(daemon)
        need(result is self.result and self.result is not None and self.observe_pending(), 'actual_inert_candidate_result_and_stopped_host_required')
        return result


def run_original_finalizer(daemon, *, stage, original_frame, continuation):
    """Join one actual original finally frame before any fallible source/DB query."""
    import inspect
    host = _HOST.get()
    need(type(host) is ReportOnlyNativeHost and host.daemon is daemon
         and getattr(daemon, '_report_only_native_host', None) is host
         and host.bound and not host.finished and host.pid == os.getpid()
         and host.thread == threading.get_ident(), 'actual_original_finalizer_host_binding_required')
    caller = inspect.currentframe().f_back
    host.frames.append((stage, original_frame, continuation, caller))
    try:
        host.finalizers.register(stage, continuation, caller, original_frame)
    except BaseException as error:
        host._retain(error)
    return host._run_original_finalizer(daemon, stage=stage, original_frame=original_frame,
        continuation=continuation, caller_frame=caller)
