"""Finite complete source capsule and private Git seed preparation.

This worker runs only in an actually attached original generic native group.
It creates inert readonly preparation outputs. It never admits the later host,
imports product modules, reconstructs grants or publishes a source change.
"""
import importlib.util
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

from fixture_io import canonical, fsync_dir, need, read, sha, source_guard, strict, wire, write_new

HERE = Path(__file__).absolute().parent
GATE = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01/repository-fixture-v02/parent_attachment_gate.py')
GATE_SHA = 'dba5662443619358e2225d0cf7ccaa2e2b6c5bf51b0a60d2d1809004d1667fb3'
DAEMON = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
ADMIN = 'ipfs_accelerate_py/agent_supervisor/runtime/native_admin_execution.py'
HOST = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_native_host.py'
ENTRY = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_disposition_gate.py'
HOOKS = {
    DAEMON,
    'ipfs_accelerate_py/agent_implementation_route.py',
    'ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py',
    'ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py',
}
EXPECTED_OVERRIDES = {ADMIN, HOST, ENTRY}


def relative(value):
    need(type(value) is str, 'string_payload_path_required')
    path = PurePosixPath(value)
    need(value and path.as_posix() == value and not path.is_absolute()
         and not any(part in {'', '.', '..', '__pycache__'} for part in path.parts)
         and not any(c in value for c in '\x00\r\n\t'), 'canonical_payload_path_required')
    return path


def population(root, *, maximum_files, maximum_bytes, readonly):
    files, directories, total = {}, [], 0
    for base, dirs, names in os.walk(root, followlinks=False):
        base = canonical(base); metadata = base.stat()
        need(metadata.st_uid == os.getuid() and stat.S_ISDIR(metadata.st_mode)
             and (not readonly or stat.S_IMODE(metadata.st_mode) == 0o500),
             'owned_payload_directory_mode_required')
        directories.append(base)
        need(len(directories) <= 4096, 'bounded_payload_directories_required')
        for name in dirs:
            need(not (base / name).is_symlink(), 'payload_directory_alias_refused')
        for name in names:
            path = canonical(base / name); item = path.lstat()
            need(stat.S_ISREG(item.st_mode) and item.st_nlink == 1
                 and item.st_uid == os.getuid()
                 and (not readonly or stat.S_IMODE(item.st_mode) == 0o444),
                 'owned_single_link_payload_file_required')
            total += item.st_size
            need(len(files) < maximum_files and total <= maximum_bytes,
                 'bounded_payload_population_required')
            raw = read(path)
            files[path.relative_to(root).as_posix()] = {'bytes': len(raw), 'sha256': sha(raw)}
    return files, directories, total


def freeze_tree(root, *, maximum_files, maximum_bytes):
    files, directories, total = population(root, maximum_files=maximum_files,
        maximum_bytes=maximum_bytes, readonly=False)
    for name in files:
        path = root / name
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fchmod(fd, 0o444); os.fsync(fd)
        finally:
            os.close(fd)
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        os.chmod(path, 0o500); fsync_dir(path)
    observed, _, observed_total = population(root, maximum_files=maximum_files,
        maximum_bytes=maximum_bytes, readonly=True)
    need(observed == files and observed_total == total, 'frozen_payload_bytes_changed')
    return files, total


def main():
    need(len(sys.argv) == 3 and sys.dont_write_bytecode,
         'exact_preparation_request_SHA_and_no_bytecode_required')
    request = strict(read(sys.argv[1], sys.argv[2], maximum=1024**2))
    need(type(request) is dict and set(request) == {
        'schema', 'parent_request_path', 'parent_request_name', 'attempt_directory',
        'storage_bytes', 'source_manifest_sha256', 'base_capsule_root',
        'base_capsule_manifest_sha256', 'base_preparation_root_observation_path',
        'base_preparation_root_observation_sha256', 'overrides',
        'hook_assembler_path', 'hook_assembler_sha256',
        'hook_assembly_report_path', 'hook_assembly_report_sha256',
        'expected_payload_file_count', 'expected_payload_bytes',
        'reviewed_composed_daemon_sha256', 'reviewed_native_admin_sha256',
        'reviewed_host_sha256', 'reviewed_entry_gate_sha256',
    } and request['schema'] == 'legal-ir-full-host-bootstrap-preparation@1'
         and request['storage_bytes'] == 128 * 1024**2,
         'closed_128MiB_preparation_request_required')
    source_guard(HERE, request['source_manifest_sha256'])
    attempt = canonical(request['attempt_directory'])
    need(attempt == canonical(os.environ['TMPDIR']) and attempt.is_dir()
         and attempt.stat().st_uid == os.getuid() and attempt.stat().st_mode & 0o077 == 0,
         'actual_private_original_worker_attempt_required')
    read(GATE, GATE_SHA)
    spec = importlib.util.spec_from_file_location('actual_full_host_preparation_attachment_gate', GATE)
    need(spec is not None and spec.loader is not None, 'original_gate_loader_required')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    attachment = gate.wait_for_original_generic_parent(request['parent_request_path'],
        attempt_directory=attempt, expected_request_name=request['parent_request_name'],
        storage_bytes=request['storage_bytes'],
        worker_command=[sys.executable, '-B', str(Path(__file__).absolute()), sys.argv[1], sys.argv[2]],
        timeout_seconds=60)
    need(attachment is not None, 'actual_original_preparation_attachment_required')
    os.umask(0o077)
    need(not any(name == 'ipfs_accelerate_py' or name.startswith('ipfs_accelerate_py.')
                 for name in sys.modules), 'no_product_import_preparation_required')
    base = canonical(request['base_capsule_root'])
    observed = strict(read(request['base_preparation_root_observation_path'],
        request['base_preparation_root_observation_sha256'], maximum=1024**2))
    need(observed.get('schema') == 'legal-ir-disposition-entry-capsule-root-observation@1'
         and observed.get('passed') is True and observed.get('native_finalized') is True
         and observed.get('actual_parent_finalized') is True
         and observed.get('actual_root_phase_owner_rows_drained') is True
         and observed.get('capsule_root') == str(base)
         and observed.get('capsule_manifest_sha256') == request['base_capsule_manifest_sha256'],
         'actual_finalized_base_capsule_join_required')
    manifest = strict(read(base / 'SOURCE_MANIFEST.json', request['base_capsule_manifest_sha256'],
                           maximum=4 * 1024**2))
    need(manifest.get('schema') == 'legal-ir-disposition-entry-capsule@1'
         and manifest.get('payload_file_count') == 2812
         and manifest.get('payload_bytes') == 106943056
         and manifest.get('capsule_root') == str(base), 'exact_complete_base_capsule_required')
    originals, sources = {}, {}
    for row in manifest['files']:
        name = relative(row['path']).as_posix()
        need(name.endswith('.py') and name not in originals, 'unique_python_source_required')
        path = canonical(base / name)
        need(stat.S_IMODE(path.stat().st_mode) == 0o444, 'readonly_base_source_required')
        raw = read(path, row['sha256'])
        need(len(raw) == row['bytes'], 'base_source_size_changed')
        originals[name] = {'bytes': len(raw), 'sha256': sha(raw)}; sources[name] = path
    need(len(originals) == 2812 and sum(row['bytes'] for row in originals.values()) == 106943056,
         'complete_base_source_population_required')
    actual, _, _ = population(base, maximum_files=2813, maximum_bytes=128 * 1024**2, readonly=True)
    need(set(actual) == set(originals) | {'SOURCE_MANIFEST.json'}, 'base_capsule_extra_or_missing_file')
    # Materialize complete modules only inside this actually charged worker.
    # This imports the pinned source assembler, never the product it assembles.
    assembler_path = canonical(request['hook_assembler_path'])
    read(assembler_path, request['hook_assembler_sha256'])
    assembly = strict(read(request['hook_assembly_report_path'],
        request['hook_assembly_report_sha256'], maximum=1024**2))
    need(assembly.get('schema') == 'legal-ir-native-bootstrap-hook-source-assembly@1'
         and assembly.get('runtime_enabled') is False,
         'reviewed_hook_source_assembly_required')
    reviewed = {row['path']: row for row in assembly['modules']}
    need(set(reviewed) == HOOKS and len(assembly['modules']) == 4,
         'exact_four_reviewed_hook_outputs_required')
    spec = importlib.util.spec_from_file_location('actual_charged_bootstrap_source_assembler', assembler_path)
    need(spec is not None and spec.loader is not None, 'source_assembler_loader_required')
    assembler = importlib.util.module_from_spec(spec); spec.loader.exec_module(assembler)
    _original, _base, proposed = assembler.assemble()
    need(type(proposed) is dict and set(proposed) == HOOKS,
         'exact_source_assembler_module_outputs_required')
    assembled_total = 0
    for name, source in proposed.items():
        need(type(source) is str, 'plain_assembled_source_required')
        raw = source.encode(); assembled_total += len(raw)
        need(len(raw) == reviewed[name]['proposed_bytes']
             and sha(raw) == reviewed[name]['proposed_sha256'],
             'reviewed_assembled_module_identity_required')
    need(assembled_total == 3807556, 'exact_charged_four_module_bytes_required')
    assembled = attempt / 'assembled'; assembled.mkdir(mode=0o700)
    for name, source in sorted(proposed.items()):
        path = assembled / relative(name); path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        write_new(path, source.encode(), mode=0o444); sources[name] = path
    overrides = request['overrides']
    need(type(overrides) is list and len(overrides) == len(EXPECTED_OVERRIDES),
         'exact_three_physical_runtime_pin_successors_required')
    seen = set()
    for row in overrides:
        need(type(row) is dict and set(row) == {'path', 'source_path', 'bytes', 'sha256'},
             'closed_source_override_required')
        name = relative(row['path']).as_posix()
        need(name in EXPECTED_OVERRIDES and name not in seen, 'exact_unique_override_required')
        path = canonical(row['source_path']); raw = read(path, row['sha256'])
        need(len(raw) == row['bytes'], 'override_source_size_changed')
        sources[name] = path; seen.add(name)
    need(seen == EXPECTED_OVERRIDES, 'all_actual_source_successors_required')
    for name, field in ((DAEMON, 'reviewed_composed_daemon_sha256'),
                        (ADMIN, 'reviewed_native_admin_sha256'),
                        (HOST, 'reviewed_host_sha256'), (ENTRY, 'reviewed_entry_gate_sha256')):
        read(sources[name], request[field])
    rows, total = [], 0
    for name, source in sorted(sources.items()):
        raw = read(source); total += len(raw)
        rows.append({'path': name, 'bytes': len(raw), 'sha256': sha(raw)})
    need(len(rows) == request['expected_payload_file_count'] == 2813
         and total == request['expected_payload_bytes'] and total <= 112 * 1024**2,
         'reviewed_complete_payload_plus_seed_headroom_required')
    need(total + assembled_total + 1024**2 + 8 * 1024**2 <= request['storage_bytes'],
         'complete_payload_assembled_modules_seed_and_metadata_must_fit_128MiB')
    capsule = attempt / 'capsule'; capsule.mkdir(mode=0o700)
    for row in rows:
        path = capsule / row['path']; path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        write_new(path, read(sources[row['path']], row['sha256']), mode=0o444)
    composed = dict(schema='legal-ir-native-bootstrap-full-host-capsule@1', capsule_root=str(capsule),
        files=rows, payload_file_count=len(rows), payload_bytes=total,
        base_manifest_sha256=request['base_capsule_manifest_sha256'],
        reviewed_composed_daemon_sha256=request['reviewed_composed_daemon_sha256'],
        reviewed_native_admin_sha256=request['reviewed_native_admin_sha256'],
        reviewed_host_sha256=request['reviewed_host_sha256'],
        reviewed_entry_gate_sha256=request['reviewed_entry_gate_sha256'],
        runtime_enabled=False, acceptance_authority=False)
    write_new(capsule / 'SOURCE_MANIFEST.json', wire(composed), mode=0o444)
    freeze_tree(capsule, maximum_files=2814, maximum_bytes=120 * 1024**2)
    seed = attempt / 'git-seed'; seed.mkdir(mode=0o700)
    environment = dict(PATH='/usr/bin:/bin', HOME=os.environ['HOME'],
        LANG='C.UTF-8', LC_ALL='C.UTF-8', GIT_CONFIG_NOSYSTEM='1',
        GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0')
    command_prefix = ['/usr/bin/git', '-c', 'core.hooksPath=/dev/null',
        '-c', 'init.templateDir=', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
        '-c', 'core.fsmonitor=false', '-c', 'user.name=Legal IR engineering fixture',
        '-c', 'user.email=fixture@example.invalid']
    commands = []
    def git(*arguments):
        result = subprocess.run(command_prefix + list(arguments), cwd=seed, env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False)
        need(result.returncode == 0 and len(result.stdout) + len(result.stderr) <= 1024**2,
             'finite_private_seed_Git_setup_failed')
        commands.append(list(arguments)); return result.stdout
    git('init', '-b', 'fixture-main')
    write_new(seed / 'tasks.todo.md', b'# Private full-host engineering fixture\n\nNo canonical task selection is authorized.\n')
    git('add', '--', 'tasks.todo.md')
    git('commit', '--no-gpg-sign', '-m', 'Private full-host qualification seed')
    head = git('rev-parse', '--verify', 'HEAD').decode('ascii').strip()
    need(len(head) == 40 and all(c in '0123456789abcdef' for c in head)
         and git('status', '--porcelain=v1', '-z') == b''
         and not (seed / '.git/objects/info/alternates').exists()
         and not (seed / '.gitmodules').exists(), 'self_contained_clean_private_seed_required')
    seed_files, seed_total = freeze_tree(seed, maximum_files=128, maximum_bytes=1024**2)
    seed_manifest = dict(schema='legal-ir-native-bootstrap-private-Git-seed@1',
        seed_root=str(seed), files=[dict(path=name, **row) for name, row in sorted(seed_files.items())],
        payload_file_count=len(seed_files), payload_bytes=seed_total, head=head,
        branch='fixture-main', file_mode='0444', directory_mode='0500',
        setup_commands=commands, alternates=False, hardlinks=False, symlinks=False,
        submodules=False, source_only_inert_input=True, acceptance_authority=False)
    write_new(attempt / 'GIT_SEED_MANIFEST.json', wire(seed_manifest), mode=0o444)
    result = dict(schema='legal-ir-native-bootstrap-preparation-result@1', passed=True,
        capsule_root=str(capsule), capsule_manifest_sha256=sha(read(capsule / 'SOURCE_MANIFEST.json')),
        capsule_payload_file_count=len(rows), capsule_payload_bytes=total,
        seed_root=str(seed), seed_manifest_sha256=sha(read(attempt / 'GIT_SEED_MANIFEST.json')),
        seed_payload_file_count=len(seed_files), seed_payload_bytes=seed_total,
        charged_assembled_module_bytes=assembled_total,
        reviewed_worst_case_bytes=total + assembled_total + 1024**2 + 8 * 1024**2,
        private_seed_head=head, actual_original_parent_attachment=attachment,
        native_parent_finalization_pending=True, full_native_host_qualified=False,
        runtime_enabled=False, acceptance_authority=False, completion_authority=False)
    write_new(attempt / 'full-host-preparation.json', wire(result))
    fsync_dir(attempt)
    print(wire(result).decode(), flush=True)


if __name__ == '__main__':
    main()
