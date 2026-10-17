"""Finite genuine complete daemon constructor/entry negative qualification.

Run only under original V05's actual admitted and attached 1CPU/4096MiB/8-slot
worker group. This creates a tiny private Git repository and private real DBs.
It never binds or constructs a native host, alters a cap, executes a provider,
creates a scheduler root, selects a task or calls original productive run_once.
"""
from dataclasses import asdict
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from fixture_io import canonical, fsync_dir, need, read, sha, source_guard, strict, wire, write_new

HERE = Path(__file__).absolute().parent
TODO = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.'
GATE = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01/repository-fixture-v02/parent_attachment_gate.py')
GATE_SHA = 'dba5662443619358e2225d0cf7ccaa2e2b6c5bf51b0a60d2d1809004d1667fb3'


def ready_capsule(request):
    capsule = canonical(request['capsule_root'])
    observed = strict(read(request['preparation_root_observation_path'],
                          request['preparation_root_observation_sha256'], maximum=1024**2))
    need(observed.get('schema') == 'legal-ir-full-host-capsule-root-observation@1'
         and observed.get('passed') is True and observed.get('actual_parent_finalized') is True
         and observed.get('actual_root_phase_owner_rows_drained') is True
         and observed.get('capsule_root') == str(capsule)
         and observed.get('capsule_manifest_sha256') == request['capsule_manifest_sha256'],
         'independent_actual_preparation_parent_join_required')
    manifest = strict(read(capsule / 'SOURCE_MANIFEST.json', request['capsule_manifest_sha256'], maximum=4 * 1024**2))
    need(manifest.get('schema') == 'legal-ir-full-host-capsule@1'
         and manifest.get('capsule_root') == str(capsule) and manifest.get('runtime_enabled') is False
         and manifest.get('base_manifest_sha256') == '71053aea3f7e5917cd218bcfbf6458fe351fb7fb36de64ed723789b0a44cee2d'
         and 2807 <= manifest.get('payload_file_count', 0) <= 2813
         and 0 < manifest.get('payload_bytes', 0) <= 128 * 1024**2
         and manifest.get('reviewed_composed_daemon_sha256') == request['reviewed_composed_daemon_sha256'],
         'actual_reviewed_complete_capsule_required')
    seen, total, pins = set(), 0, {}
    for row in manifest['files']:
        name = row['path']
        path = canonical(capsule / name)
        need(capsule in path.parents and name.endswith('.py') and name not in seen
             and path.stat().st_mode & 0o222 == 0, 'unique_readonly_python_capsule_file_required')
        raw = read(path, row['sha256'])
        need(len(raw) == row['bytes'], 'capsule_source_bytes_changed')
        pins[str(path)] = row['sha256']; seen.add(name); total += len(raw)
    need(len(seen) == manifest['payload_file_count'] and total == manifest['payload_bytes'], 'complete_capsule_population_required')
    observed_files = set()
    for directory, dirs, files in os.walk(capsule, followlinks=False):
        need(canonical(directory).stat().st_mode & 0o222 == 0, 'readonly_capsule_directory_required')
        for name in dirs:
            need(not (Path(directory) / name).is_symlink(), 'capsule_directory_alias_refused')
        for name in files:
            observed_files.add((Path(directory) / name).relative_to(capsule).as_posix())
    need(observed_files == seen | {'SOURCE_MANIFEST.json'}, 'capsule_has_unreviewed_extra_files')
    return capsule, pins


def main():
    need(len(sys.argv) == 3, 'reviewed_missing_host_request_and_SHA_required')
    request_raw = read(sys.argv[1], sys.argv[2], maximum=1024**2)
    request = strict(request_raw)
    need(type(request) is dict and set(request) == {'schema', 'parent_request_path', 'parent_request_name',
        'attempt_directory', 'storage_bytes', 'capsule_root', 'capsule_manifest_sha256',
        'preparation_root_observation_path', 'preparation_root_observation_sha256',
        'source_manifest_sha256', 'reviewed_composed_daemon_sha256'}
        and request['schema'] == 'legal-ir-full-original-daemon-missing-host-request@1'
        and request['storage_bytes'] == 268435456, 'closed_finite_missing_host_request_required')
    source_guard(HERE, request['source_manifest_sha256'])
    root = canonical(request['attempt_directory'])
    need(root == canonical(os.environ['TMPDIR']) and root.is_dir() and root.stat().st_uid == os.getuid()
         and root.stat().st_mode & 0o077 == 0, 'actual_private_native_worker_attempt_required')
    # Original V05 launches before attaching; no product import, Git command or
    # private DB open can occur until its parent-owned running marker joins.
    read(GATE, GATE_SHA)
    loader = importlib.util.spec_from_file_location('exact_original_parent_attachment_gate', GATE)
    need(loader is not None and loader.loader is not None, 'exact_attachment_gate_loader_required')
    gate = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(gate)
    observation = gate.wait_for_original_generic_parent(request['parent_request_path'],
        attempt_directory=root, expected_request_name=request['parent_request_name'],
        storage_bytes=request['storage_bytes'],
        worker_command=[sys.executable, '-B', str(Path(__file__).absolute()), sys.argv[1], sys.argv[2]],
        timeout_seconds=60)
    need(observation is not None, 'original_attached_worker_ordering_observation_required')
    capsule, pins = ready_capsule(request)
    need(not any(name == 'ipfs_accelerate_py' or name.startswith('ipfs_accelerate_py.')
                 for name in sys.modules), 'fresh_complete_product_import_required')
    for name in tuple(os.environ):
        if name.startswith(('IPFS_ACCELERATE_AGENT_', 'GIT_')):
            del os.environ[name]
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', IPFS_ACCEL_SKIP_CORE='1',
        IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_AUTO_INSTALL='false', IPFS_DATASETS_AUTO_INSTALL='false',
        IPFS_ACCELERATE_DUCKDB_ONLY='1', IPFS_ACCELERATE_AGENT_QUACK_PREFER='false',
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(root / 'orchestration'),
        IPFS_ACCELERATE_AGENT_RECLAIM_DEAD_WORKTREE_LEASES_ON_STARTUP='false',
        GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    need(sys.getprofile() is None, 'unoccupied_original_worker_profiler_required')
    phase, processes = ['private_git_setup'], []

    def observe_git_only(frame, event, _value):
        if frame.f_code is not subprocess.Popen.__init__.__code__:
            return
        if event == 'call':
            args = frame.f_locals['args']
            need(type(args) in {list, tuple} and args and args[0] in {'git', '/usr/bin/git'}
                 and not frame.f_locals.get('shell') and not frame.f_locals.get('start_new_session')
                 and frame.f_locals.get('process_group') is None,
                 'only_original_local_Git_in_attached_worker_group_allowed')
            need(len(processes) < 64, 'finite_original_Git_process_count_required')
        elif event == 'return':
            child = frame.f_locals['self']
            need(type(child.pid) is int and os.getpgid(child.pid) == os.getpgrp(), 'original_Git_escaped_attached_worker_group')
            processes.append({'phase': phase[0], 'pid': child.pid, 'pgid': os.getpgrp()})

    sys.setprofile(observe_git_only)
    try:
        repo = root / 'repository'
        repo.mkdir(mode=0o700)
        git_prefix = ['/usr/bin/git', '-c', 'core.hooksPath=/dev/null', '-c', 'gc.auto=0',
            '-c', 'user.name=Legal IR engineering fixture', '-c', 'user.email=fixture@example.invalid']
        def git(*arguments):
            result = subprocess.run(git_prefix + list(arguments), cwd=repo, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False)
            need(result.returncode == 0 and len(result.stdout) + len(result.stderr) <= 1024**2,
                 'private_fixture_Git_setup_failed')
        git('init', '-b', 'fixture-main')
        write_new(repo / 'tasks.todo.md', b'# Private engineering fixture\n\nNo canonical task selection is authorized.\n')
        git('add', '--', 'tasks.todo.md')
        git('commit', '--no-gpg-sign', '-m', 'Private missing-host qualification fixture')
        phase[0] = 'complete_original_module_import'
        sys.path.insert(0, str(capsule))
        daemon_module = importlib.import_module(TODO + 'implementation_daemon')
        host_module = importlib.import_module(TODO + 'report_only_native_host')
        control = importlib.import_module(TODO + 'portal_task_state_control_plane')
        state_module = importlib.import_module('ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state')
        for module in (daemon_module, host_module, control, state_module):
            path = str(Path(module.__file__).absolute())
            need(path in pins, 'actual_complete_capsule_module_origin_required')
            read(path, pins[path])
        daemon_path = str(Path(daemon_module.__file__).absolute())
        need(pins[daemon_path] == request['reviewed_composed_daemon_sha256'] == host_module.DAEMON_SHA,
             'exact_latest_composed_daemon_and_host_required')
        lane = root / 'lane'
        lane.mkdir(mode=0o700)
        state_path, target = lane / 'state.json', lane / 'state.duckdb'
        phase[0] = 'complete_original_constructor'
        daemon = daemon_module.PortalImplementationDaemon(
            todo_path=repo / 'tasks.todo.md', state_path=state_path,
            strategy_path=lane / 'strategy.json', events_path=lane / 'events.jsonl',
            repo_root=repo, implement=True, implementation_command='/usr/bin/false',
            production_provider_policy='', max_task_attempts=2, preserve_task_attempts=True,
            use_ephemeral_worktree=True, worktree_root=root / 'worktrees',
            retain_worktree_artifacts=True, worktree_pool_enabled=False,
            merge_target_branch='fixture-main', merge_queue_dir=root / 'merge-queue',
            validation_cache_dir=root / 'validation-cache', validation_max_workers=1,
            validation_resource_budget=1, proof_cache_dir=root / 'proof-cache',
            maintenance_interval_seconds=0, merge_reconciliation_max_merges=0,
            merged_worktree_cleanup_max=0, require_active_plan_revision=False,
            worktree_submodule_paths=(), report_only_candidate_freezing=True)
        need(type(daemon) is daemon_module.PortalImplementationDaemon and daemon.max_task_attempts == 2
             and daemon._report_only_native_host is None and host_module._HOST.get() is None,
             'actual_original_constructor_unbound_host_default_required')
    finally:
        sys.setprofile(None)
    connection = state_module.open_duckdb_connection(target, prefer_quack=False, memory_limit='64MB', threads=1)
    need(type(connection) is state_module.DuckDBConnection, 'actual_original_private_connection_required')
    try:
        connection.execute(control.DDL)
        connection.execute(control.HANDOFF_DDL)
        initial = daemon_module.PortalTaskState(active_attempt=2,
            implementation_attempts={'fixture-only': 2}, completed_task_ids=[])
        connection.execute('INSERT INTO portal_task_state VALUES (?, ?, ?, FALSE, ?)',
            [state_path.stem, json.dumps(asdict(initial), sort_keys=True), 1, 'fixture-baseline'])
        control.bind_task_state_control_plane(state_path, str(target))
        before = connection.execute('SELECT * FROM portal_task_state ORDER BY lane_key').fetchall()
        original_raw, epoch = connection._connection, connection._reconnect_epoch
        forbidden = {getattr(daemon_module.PortalImplementationDaemon, name).__code__ for name in (
            '_run_once', '_retained_provider_capacity_deferral', '_mark_implementation_started')}
        forbidden.update({control._connect.__code__, control._project_ducklake.__code__,
            state_module.open_duckdb_connection.__code__, state_module.DuckDBConnection.close.__code__,
            state_module.DuckDBConnection.reconnect_exclusive_owner.__code__, subprocess.Popen.__init__.__code__,
            state_module.DuckDBConnection.execute.__code__})
        require_calls = []
        def observe_refusal(frame, event, _value):
            if event == 'call' and frame.f_code in forbidden:
                raise AssertionError('missing_host_entered_original_productive_or_database_body')
            if event == 'call' and frame.f_code is host_module.require_report_only_native_host.__code__:
                need(frame.f_locals['daemon'] is daemon, 'actual_full_daemon_identity_changed')
                require_calls.append(True)
        observed = None
        sys.setprofile(observe_refusal)
        try:
            daemon.run_once()
        except host_module.NativeHostRefusal as error:
            observed = error
        finally:
            sys.setprofile(None)
        need(type(observed) is host_module.NativeHostRefusal
             and str(observed) == 'explicit_same_process_native_host_required' and len(require_calls) == 1,
             'exact_original_missing_host_entry_refusal_required')
        need(connection.execute('SELECT * FROM portal_task_state ORDER BY lane_key').fetchall() == before
             and daemon.max_task_attempts == 2 and connection._connection is original_raw
             and connection._reconnect_epoch == epoch and not connection._closed
             and daemon._report_only_native_host is None and host_module._HOST.get() is None,
             'missing_host_changed_counter_cap_connection_or_binding')
        need(connection.execute('SELECT 61').fetchone()[0] == 61, 'fixture_caller_handle_not_alive')
    finally:
        connection.close()
    source_guard(HERE, request['source_manifest_sha256'])
    ready_capsule(request)
    read(sys.argv[1], sys.argv[2], maximum=1024**2)
    result = {'schema': 'legal-ir-full-original-daemon-missing-host-result@1', 'passed': True,
        'source_manifest_sha256': request['source_manifest_sha256'], 'request_sha256': sys.argv[2],
        'capsule_manifest_sha256': request['capsule_manifest_sha256'],
        'reviewed_composed_daemon_sha256': request['reviewed_composed_daemon_sha256'],
        'full_original_module_imported': True, 'full_original_daemon_constructed': True,
        'actual_private_original_DuckDB_wrapper': True, 'actual_missing_host_refusal_observed': True,
        'original_cap_two_unchanged': True, 'original_private_counter_row_unchanged': True,
        'actual_native_host_constructed': False, 'genuine_cap_two_host_refusal_qualified': False,
        'same_borrow_connection_identity_qualified': False, 'full_host_qualified': False,
        'original_Git_process_count': len(processes), 'original_Git_processes': processes,
        'Git_descendants_in_original_attached_worker_group_observed': True,
        'native_parent_finalization_claimed': False, 'provider_calls_performed': False,
        'task_selection_performed': False, 'canonical_attempt_counter_writes': False,
        'canonical_tasks_accepted': 0, 'production_hook_deployed': False,
        'acceptance_authority': False, 'completion_authority': False}
    write_new(root / 'full-daemon-missing-host.json', wire(result))
    fsync_dir(root)
    print(wire(result).decode(), end='', flush=True)


if __name__ == '__main__':
    main()
