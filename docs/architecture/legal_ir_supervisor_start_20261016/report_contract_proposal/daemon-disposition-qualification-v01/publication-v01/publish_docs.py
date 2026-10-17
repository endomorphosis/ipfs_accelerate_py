"""Publish one exact additive documentation generation on the owned draft.

The original generic native parent owns all admission and group cleanup. This
worker installs no runtime and performs no CI/GitHub qualification or merge.
"""
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import selectors
import stat
import subprocess
import sys
import time

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
REPO = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
BASE = 'a1b1f32826e58913837308330d8ec6cc8589222a'
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
REMOTE = 'https://github.com/endomorphosis/ipfs_accelerate_py.git'
OLD_PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016/report_contract_proposal'
PREFIX = OLD_PREFIX + '/daemon-disposition-qualification-v01'
RUNTIME = REPO / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
RUNTIME_SHA = '23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed'
BOARD = REPO / 'docs/architecture/legal_ir_supervisor_start_20261016/tasks.todo.md'
BOARD_SHA = '5244288f69562663f44b00d27add1ee6ef338f977fed1c0e1c8855725ff372c7'
GATE = HERE / 'parent_attachment_gate.py'
GATE_SHA = '18fe92a023468a1d81f147681640cbbec8b67fba3dc745a64b2af536576f9cf6'
DEADLINE = None


def need(value, reason):
    if not value: raise RuntimeError(reason)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(path, maximum=16 * 1024**2):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased_owned_path_required')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and not before.st_mode & 2 and 0 <= before.st_size <= maximum, 'bounded_owned_regular_file_required')
        raw = stream.read(maximum + 1); after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns, row.st_mode, row.st_nlink)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'original_source_changed')
    return raw


def digest_file(path, maximum=64 * 1024**2):
    """Bounded streaming tool/source hash, including the 45 MB gh binary."""
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased_input_pin_required')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and not before.st_mode & 2 and 0 <= before.st_size <= maximum, 'bounded_input_pin_required')
        digest = hashlib.sha256(); size = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(block); need(size <= before.st_size, 'input_pin_grew'); digest.update(block)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(size == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'input_pin_changed')
    return digest.hexdigest()


def relative(value):
    need(type(value) is str, 'relative_document_name_required')
    path = PurePosixPath(value)
    need(value and not path.is_absolute() and not any(p in {'', '.', '..', '.git', '__pycache__'} for p in path.parts)
         and path.as_posix() == value and not any(c in value for c in '\x00\r\n\t'), 'canonical_relative_document_required')
    return path


def command(argv, *, input=None, limit=32 * 1024**2):
    need(time.monotonic() < DEADLINE, 'finite_publication_deadline_reached')
    child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL if input is None else subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_CONFIG_NOSYSTEM='1',
                 GIT_OPTIONAL_LOCKS='0', GIT_PAGER='cat', GH_PAGER='cat'))
    # All children inherit the actual originally admitted worker group. If a
    # stream or finite command fails, the outer owner reaps that complete group.
    streams = {'stdout': bytearray(), 'stderr': bytearray()}
    until = min(DEADLINE, time.monotonic() + 180)
    with selectors.DefaultSelector() as selector:
        for name in streams:
            stream = getattr(child, name); os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        remaining = memoryview(input) if input is not None else None
        if remaining is not None:
            os.set_blocking(child.stdin.fileno(), False)
            selector.register(child.stdin, selectors.EVENT_WRITE, 'stdin')
        while selector.get_map() or child.poll() is None:
            need(time.monotonic() < until, 'finite_publication_command_deadline_reached')
            for key, _ in selector.select(.1):
                if key.data == 'stdin':
                    if remaining:
                        try: remaining = remaining[os.write(key.fileobj.fileno(), remaining[:65536]):]
                        except BrokenPipeError: remaining = memoryview(b'')
                    if not remaining:
                        selector.unregister(key.fileobj); key.fileobj.close()
                    continue
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    selector.unregister(key.fileobj); key.fileobj.close(); continue
                streams[key.data].extend(block)
                need(sum(map(len, streams.values())) <= limit, 'bounded_publication_command_output_required')
    code = child.wait()
    # Avoid exposing credential-bearing error text from Git/gh helpers.
    need(code == 0, 'publication_command_failed:' + str(argv[0]) + ':' + str(code))
    return bytes(streams['stdout'])


def git(*args, input=None):
    return command(['/usr/bin/git', '-c', 'core.hooksPath=/dev/null', '-c', 'gc.auto=0',
                    '-c', 'maintenance.auto=false', '-c', 'core.fsmonitor=false',
                    '-c', 'pack.threads=1', '-c', 'pack.windowMemory=64m',
                    '-c', 'pack.deltaCacheSize=32m', *args], input=input)


def blob_oid(raw):
    return hashlib.sha1(('blob ' + str(len(raw)) + '\0').encode() + raw).hexdigest()


def existing_inventory():
    proof_raw = read(HERE / 'EXISTING_REPORT_INVENTORY.json')
    proof = json.loads(proof_raw)
    need(proof['base_head'] == BASE and proof['prefix'] == OLD_PREFIX
         and proof['file_count'] == len(proof['files']) == 593
         and proof['total_bytes'] == 16547750, 'exact_existing_generation_source_inventory_required')
    rows = []
    for row in proof['files']:
        name = relative(row['path']).as_posix()
        need(name.startswith(OLD_PREFIX + '/') and not name.startswith(PREFIX + '/'), 'only_existing_generations_allowed')
        raw = read(REPO / name)
        need(len(raw) == row['bytes'] and sha(raw) == row['sha256'] and blob_oid(raw) == row['blob_oid'],
             'existing_document_bytes_changed')
        rows.append(b'100644 blob ' + row['blob_oid'].encode() + b'\t' + name.encode() + b'\0')
    expected = b''.join(rows)
    need(sha(expected) == proof['git_tree_raw_sha256'], 'exact_existing_inventory_encoding_required')
    need(git('ls-tree', '-r', '-z', '--full-tree', BASE, '--', OLD_PREFIX) == expected,
         'existing_generations_not_exact_committed_base_tree')
    return expected, sha(proof_raw)


def write(path, raw):
    need(not path.exists() and not any(p.is_symlink() for p in (path, *path.parents)), 'fresh_additive_document_required')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o644); stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def flush_external_outputs():
    """Flush exactly new documentation and original local Git metadata.

    Object alternates are ordinary path text here; never traverse their targets.
    The reviewed outer request precharges these two external roots explicitly.
    """
    count = 0; size = 0; directories = set()
    for root, limit in ((REPO / PREFIX, 16 * 1024**2), (REPO / '.git', 64 * 1024**2)):
        local = 0
        for base, dirs, files in os.walk(root, followlinks=False):
            directories.add(Path(base))
            need(not any((Path(base) / name).is_symlink() for name in dirs), 'external_directory_alias_refused')
            for name in files:
                path = Path(base) / name; metadata = path.lstat()
                need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and metadata.st_uid == os.getuid(),
                     'external_git_or_document_file_alias_refused')
                count += 1; local += metadata.st_size; size += metadata.st_size
                need(count <= 4096 and local <= limit, 'explicit_external_document_or_git_charge_exceeded')
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try: os.fsync(fd)
                finally: os.close(fd)
    path = REPO / PREFIX
    while path != REPO.parent:
        directories.add(path); path = path.parent
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
    return {'flushed_files': count, 'flushed_apparent_bytes': size, 'alternate_object_stores_traversed': False}


def main():
    global DEADLINE
    need(len(sys.argv) == 3, 'packet_digest_and_original_request_path_required')
    packet_raw = read(HERE / 'PACKET.json'); need(sha(packet_raw) == sys.argv[1], 'publication_packet_changed')
    packet = json.loads(packet_raw)
    need(packet['base_head'] == BASE and packet['branch'] == BRANCH and packet['prefix'] == PREFIX
         and packet['github_qualification'] == 'skipped_by_user', 'exact_reviewed_publication_scope_required')
    need(sha(read(GATE)) == GATE_SHA, 'original_attachment_gate_source_changed')
    spec = importlib.util.spec_from_file_location('publication_original_parent_order_gate', GATE)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    request_path = Path(sys.argv[2]); request = json.loads(read(request_path))
    need(request.get('storage_bytes') == 256 * 1024**2 and request.get('duration_seconds') == 1800
         and request.get('cwd') == str(HERE), 'exact_finite_publication_request_resources_required')
    expected_external = [{'path': str(REPO / PREFIX), 'bytes': 16 * 1024**2},
                         {'path': str(REPO / '.git'), 'bytes': 64 * 1024**2}]
    need(request.get('external_charges') == expected_external, 'exact_explicit_external_publication_charges_required')
    attachment = gate.wait_for_original_generic_parent(request_path,
        attempt_directory=Path(request['attempt_directory']), expected_request_name=request['name'],
        storage_bytes=request['storage_bytes'], expected_external_charges=expected_external,
        worker_command=['/usr/bin/python3.12', '-B', str(Path(__file__).absolute()), sys.argv[1], str(request_path)])
    DEADLINE = time.monotonic() + 1200
    need(sha(read(RUNTIME)) == RUNTIME_SHA and sha(read(BOARD)) == BOARD_SHA, 'original_runtime_or_board_changed')
    source_manifest_raw = read(HERE / 'SOURCE_MANIFEST.json')
    need(sha(source_manifest_raw) == packet['publisher_source_manifest_sha256'], 'publisher_source_manifest_changed')
    source_manifest = json.loads(source_manifest_raw)
    for row in source_manifest['files']:
        raw = read(HERE / relative(row['path']))
        need(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'publisher_implementation_source_changed')
    for name, digest in source_manifest['input_pins'].items():
        need(digest_file(Path(name)) == digest, 'publisher_reviewed_input_changed')
    rows = packet['files']; need(0 < len(rows) <= 256, 'bounded_document_population_required')
    contents = {}
    for row in rows:
        name = relative(row['path']).as_posix(); need(name not in contents, 'unique_document_names_required')
        need(not name.startswith('publication-v01/PACKET') and name != 'publication-request-r01.json'
             and not name.startswith('publication-root-observation'), 'publication_input_cycle_refused')
        raw = read(ROOT / name)
        need(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'exact_document_source_required')
        contents[name] = raw
    need(sum(map(len, contents.values())) <= 8 * 1024**2, 'bounded_publication_bytes_required')
    need(git('symbolic-ref', '--short', 'HEAD').decode().strip() == BRANCH
         and git('rev-parse', 'HEAD').decode().strip() == BASE, 'exact_original_owned_branch_head_required')
    need(git('status', '--porcelain=v1', '-z', '--untracked-files=all') == b'', 'original_owned_checkout_not_clean')
    old_tree, old_proof_sha = existing_inventory()
    remote = git('ls-remote', '--heads', REMOTE, 'refs/heads/' + BRANCH).decode().strip().split()
    need(remote == [BASE, 'refs/heads/' + BRANCH], 'exact_original_remote_head_required')
    pr = json.loads(command(['/usr/bin/gh', 'api', 'repos/endomorphosis/ipfs_accelerate_py/pulls/292',
                            '--jq', '{state,draft,head_ref:.head.ref,head_sha:.head.sha}']))
    need(pr == {'state': 'open', 'draft': True, 'head_ref': BRANCH, 'head_sha': BASE}, 'original_owned_draft_PR_required')
    need(not (REPO / PREFIX).exists(), 'fresh_documentation_generation_required')
    for name, raw in contents.items(): write(REPO / PREFIX / name, raw)
    names = sorted(PREFIX + '/' + name for name in contents)
    git('add', '--sparse', '-f', '--', *names)
    expected_diff = b''.join(b'A\0' + name.encode() + b'\0' for name in names)
    need(git('diff', '--cached', '--no-renames', '--name-status', '-z') == expected_diff,
         'exact_additive_staged_document_paths_required')
    batch = git('cat-file', '--batch', input=''.join(':' + name + '\n' for name in names).encode())
    offset = 0
    for name in names:
        end = batch.index(b'\n', offset); header = batch[offset:end].split(); offset = end + 1
        need(len(header) == 3 and header[1] == b'blob', 'actual_staged_blob_required')
        size = int(header[2]); raw = batch[offset:offset + size]; offset += size
        need(batch[offset:offset + 1] == b'\n' and raw == contents[name[len(PREFIX) + 1:]], 'actual_staged_document_bytes_differ')
        offset += 1
    need(offset == len(batch), 'exact_staged_batch_population_required')
    git('-c', 'user.name=Legal IR Engineering Agent', '-c', 'user.email=legal-ir-agent@example.invalid',
        'commit', '--no-gpg-sign', '-m', 'docs: document Legal IR durable denial journals and entry guards')
    head = git('rev-parse', 'HEAD').decode().strip()
    need(git('rev-parse', 'HEAD^').decode().strip() == BASE, 'exact_single_original_commit_parent_required')
    need(git('diff-tree', '--no-commit-id', '--no-renames', '--name-status', '-r', '-z', head) == expected_diff,
         'exact_additive_committed_paths_required')
    committed_old = git('ls-tree', '-r', '-z', '--full-tree', head, '--', OLD_PREFIX)
    old_only = b''.join(row + b'\0' for row in committed_old.split(b'\0') if row
                        and not row.split(b'\t', 1)[1].startswith((PREFIX + '/').encode()))
    need(old_only == old_tree, 'preexisting_committed_generations_changed')
    need(sha(read(RUNTIME)) == RUNTIME_SHA and sha(read(BOARD)) == BOARD_SHA
         and git('status', '--porcelain=v1', '-z', '--untracked-files=all') == b'', 'runtime_board_or_owned_checkout_changed')
    git('push', REMOTE, 'HEAD:refs/heads/' + BRANCH)
    remote = git('ls-remote', '--heads', REMOTE, 'refs/heads/' + BRANCH).decode().strip().split()
    need(remote == [head, 'refs/heads/' + BRANCH], 'exact_published_owned_head_required')
    durability = flush_external_outputs()
    result = {'schema': 'legal-ir-daemon-disposition-documentation-publication@1', 'passed': True,
        'base_head': BASE, 'head': head, 'branch': BRANCH, 'draft_PR': 'https://github.com/endomorphosis/ipfs_accelerate_py/pull/292',
        'document_prefix': PREFIX, 'document_files': len(contents), 'document_bytes': sum(map(len, contents.values())),
        'packet_sha256': sys.argv[1], 'preserved_original_report_files': 593, 'existing_inventory_sha256': old_proof_sha,
        'runtime_source_changed': False, 'runtime_enabled': False, 'canonical_board_changed': False,
        'github_qualification': 'skipped_by_user', 'canonical_tasks_accepted': 0, 'acceptance_authority': False,
        'parent_attachment_order_observation': attachment, 'original_native_finalization_pending': True}
    result['external_output_durability_observation'] = durability
    attempt = Path(os.environ['TMPDIR']); raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    fd = os.open(attempt / 'publication.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()
