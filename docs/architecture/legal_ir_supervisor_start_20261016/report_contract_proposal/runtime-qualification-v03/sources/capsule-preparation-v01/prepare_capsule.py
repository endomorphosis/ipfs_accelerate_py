"""Copy one exact complete Python capsule under the genuine outer prep owner.

Source-only until root invokes the original finite native owner entry. This
worker imports no product code, starts no processes, executes no copied source
and creates no native grant. Private owner custody/finalization belongs to root.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat
import sys

HERE = Path(__file__).absolute().parent
SOURCE_ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
HOOK = Path('/home/barberb/lift_coding/artifacts/legal-ir-real-handoff-20261017-01/ordinary-success-hook-v01')
HOOK_MANIFEST = '59761cd0b0ea3395efb82ef27c6bb9e3fdbe33ce7a118afb8827bea76e2fa5b9'
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-ordinary-handoff-native-20261017-r01/preparation')
CAPSULE = ATTEMPT / 'capsule'
MAX_FILES, MAX_PAYLOAD, MAX_SOURCE = 4096, 128 * 1024 * 1024, 16 * 1024 * 1024
OVERRIDES = {'ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py',
             'ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py'}
FIELDS = {'schema', 'source_root', 'source_selection', 'tracked_python_listing_sha256', 'tracked_python_file_count',
          'tracked_python_bytes', 'payload_file_count', 'payload_bytes', 'maximum_payload_bytes',
          'maximum_file_count', 'maximum_source_file_bytes', 'frozen_hook_manifest_path',
          'frozen_hook_manifest_sha256', 'files', 'author_runtime_or_native_operations_performed',
          'provider_calls_performed', 'production_hook_deployed', 'canonical_tasks_accepted',
          'acceptance_authority', 'completion_authority'}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical unaliased absolute source/output required')
    return path


def metadata(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid, info.st_nlink)


def read_source(path, *, expected_size=None, expected_sha=None, output=None, maximum=MAX_SOURCE):
    path = canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count, chunks = hashlib.sha256(), 0, []
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 <= before.st_size <= maximum,
             'bounded stable owned regular source required')
        if expected_size is not None:
            need(type(expected_size) is int and before.st_size == expected_size, 'source size differs from frozen inventory')
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            need(count <= before.st_size, 'source grew during copy/read')
            digest.update(chunk)
            if output is not None:
                output.write(chunk)
            elif expected_sha is None:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    need(count == before.st_size and metadata(before) == metadata(after) == metadata(path.lstat()),
         'source changed during copy/read')
    if expected_sha is not None:
        need(type(expected_sha) is str and len(expected_sha) == 64 and digest.hexdigest() == expected_sha,
             'source SHA differs from frozen inventory')
    return b''.join(chunks) if expected_sha is None and output is None else {'bytes': count, 'sha256': digest.hexdigest()}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path, raw, mode=0o444):
    path = canonical(path)
    need(path.parent.is_dir() and type(raw) is bytes and len(raw) <= 4 * 1024 * 1024, 'bounded new output required')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, 'wb') as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def load_inventory(path, expected_sha):
    raw = read_source(path, maximum=4 * 1024 * 1024)
    need(sha(raw) == expected_sha, 'exact frozen source inventory SHA required')
    inventory = strict_json(raw)
    need(type(inventory) is dict and set(inventory) == FIELDS
         and inventory['schema'] == 'legal-ir-native-handoff-capsule-source-inventory@1'
         and inventory['source_root'] == str(SOURCE_ROOT)
         and inventory['source_selection'] == 'exact_git_tracked_ipfs_accelerate_py_python_files_plus_frozen_hook_overrides'
         and inventory['frozen_hook_manifest_path'] == str(HOOK / 'SOURCE_MANIFEST.json')
         and inventory['frozen_hook_manifest_sha256'] == HOOK_MANIFEST
         and inventory['maximum_payload_bytes'] == MAX_PAYLOAD and inventory['maximum_file_count'] == MAX_FILES
         and inventory['maximum_source_file_bytes'] == MAX_SOURCE, 'exact complete private Python capsule scope required')
    hook_raw = read_source(HOOK / 'SOURCE_MANIFEST.json', maximum=1024 * 1024)
    need(sha(hook_raw) == HOOK_MANIFEST, 'original frozen hook manifest changed')
    hook_rows = {row['path']: row for row in strict_json(hook_raw)['files']}
    rows = inventory['files']
    need(type(rows) is list and 1 <= len(rows) <= MAX_FILES
         and inventory['payload_file_count'] == len(rows), 'bounded exact payload population required')
    names, originals, overridden, total, tracked_total = [], [], set(), 0, 0
    for row in rows:
        need(type(row) is dict and set(row) == {'path', 'source_path', 'origin', 'bytes', 'sha256', 'tracked_preimage'},
             'closed source inventory row required')
        relative = PurePosixPath(row['path'])
        need(not relative.is_absolute() and relative.as_posix() == row['path'] and relative.suffix == '.py'
             and relative.parts[0] == 'ipfs_accelerate_py' and '..' not in relative.parts
             and '.git' not in relative.parts and '__pycache__' not in relative.parts,
             'only canonical package Python source files are permitted')
        need(type(row['bytes']) is int and 0 <= row['bytes'] <= MAX_SOURCE
             and type(row['sha256']) is str and len(row['sha256']) == 64, 'bounded exact Python source row required')
        names.append(row['path'])
        total += row['bytes']
        if row['origin'] == 'tracked_owned_controller':
            need(row['source_path'] == str(SOURCE_ROOT / row['path']) and row['tracked_preimage'] is None,
                 'exact tracked owned-controller source required')
            originals.append(row['path'])
            tracked_total += row['bytes']
        else:
            need(row['origin'] == 'frozen_hook_override' and row['path'] in OVERRIDES
                 and row['path'] not in overridden and row['source_path'] == str(HOOK / 'proposed' / row['path']),
                 'only exact original frozen runner/hook substitutions allowed')
            selected = hook_rows.get('proposed/' + row['path'])
            need(selected is not None and selected['bytes'] == row['bytes'] and selected['sha256'] == row['sha256'],
                 'actual override must match original frozen hook source')
            overridden.add(row['path'])
            preimage = row['tracked_preimage']
            if row['path'].endswith('/grok_cli_runner.py'):
                need(type(preimage) is dict and set(preimage) == {'source_path', 'bytes', 'sha256'}
                     and preimage['source_path'] == str(SOURCE_ROOT / row['path'])
                     and preimage['sha256'] == 'e613c9b63f68412d6516872054645d691dda8d54de94ad862b4ec277d7c18308',
                     'original owned runner baseline must remain independently pinned')
                originals.append(row['path'])
                tracked_total += preimage['bytes']
            else:
                need(preimage is None, 'new hook source cannot masquerade as an existing tracked preimage')
    need(names == sorted(set(names)) and overridden == OVERRIDES and total == inventory['payload_bytes'] <= MAX_PAYLOAD,
         'unique complete bounded payload required')
    original_listing = ('\0'.join(sorted(originals)) + '\0').encode()
    need(len(originals) == inventory['tracked_python_file_count']
         and tracked_total == inventory['tracked_python_bytes']
         and sha(original_listing) == inventory['tracked_python_listing_sha256'],
         'original exact tracked Python source inventory correspondence failed')
    return inventory, raw


def main():
    need(len(sys.argv) == 3, 'usage: prepare_capsule.py ABSOLUTE_SOURCE_INVENTORY EXPECTED_INVENTORY_SHA256')
    need(canonical(os.environ['TMPDIR']) == ATTEMPT, 'original genuine outer-owner private preparation attempt required')
    info = ATTEMPT.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
         'fresh private owner attempt directory required')
    inventory_path = canonical(sys.argv[1])
    need(inventory_path == HERE / 'SOURCE_INVENTORY.json', 'exact reviewed source inventory path required')
    inventory, raw_inventory = load_inventory(inventory_path, sys.argv[2])
    need(not CAPSULE.exists(), 'fresh private capsule required; never replay or overwrite')
    def guard():
        checked, raw = load_inventory(inventory_path, sys.argv[2])
        need(checked == inventory and raw == raw_inventory, 'frozen capsule inventory changed')
        for row in inventory['files']:
            read_source(row['source_path'], expected_size=row['bytes'], expected_sha=row['sha256'])
            if row['tracked_preimage'] is not None:
                preimage = row['tracked_preimage']
                read_source(preimage['source_path'], expected_size=preimage['bytes'], expected_sha=preimage['sha256'])
    guard()
    CAPSULE.mkdir(mode=0o700)
    copied = []
    for row in inventory['files']:
        destination = canonical(CAPSULE / row['path'])
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(descriptor, 'wb') as stream:
            pin = read_source(row['source_path'], expected_size=row['bytes'], expected_sha=row['sha256'], output=stream)
            os.fchmod(stream.fileno(), 0o444)
            stream.flush()
            os.fsync(stream.fileno())
        read_source(destination, expected_size=row['bytes'], expected_sha=row['sha256'])
        copied.append({'path': row['path'], **pin})
    guard()
    manifest = {'schema': 'legal-ir-native-handoff-capsule@1', 'capsule_root': str(CAPSULE),
        'source_inventory_sha256': sha(raw_inventory), 'frozen_hook_manifest_sha256': HOOK_MANIFEST,
        'source_selection': inventory['source_selection'],
        'tracked_python_listing_sha256': inventory['tracked_python_listing_sha256'],
        'tracked_python_file_count': inventory['tracked_python_file_count'], 'payload_file_count': len(copied),
        'payload_bytes': sum(row['bytes'] for row in copied), 'maximum_payload_bytes': MAX_PAYLOAD,
        'maximum_file_count': MAX_FILES, 'files': copied,
        'provider_calls_performed': False, 'production_hook_deployed': False, 'canonical_tasks_accepted': 0,
        'acceptance_authority': False, 'completion_authority': False}
    manifest_raw = wire(manifest)
    write_new(CAPSULE / 'SOURCE_MANIFEST.json', manifest_raw)
    directories = [Path(directory) for directory, _names, _files in os.walk(CAPSULE, followlinks=False)]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        canonical(directory)
        fsync_directory(directory)
        directory.chmod(0o500)
    fsync_directory(ATTEMPT)
    guard()
    outcome = {'schema': 'legal-ir-native-handoff-capsule-preparation@1', 'passed': True,
        'capsule_root': str(CAPSULE), 'capsule_manifest_path': str(CAPSULE / 'SOURCE_MANIFEST.json'),
        'capsule_manifest_sha256': sha(manifest_raw), 'source_inventory_sha256': sha(raw_inventory),
        'payload_file_count': len(copied), 'payload_bytes': manifest['payload_bytes'],
        'all_files_fsynced_readonly': True, 'original_native_preparation_closeout_pending': True,
        'two_owner_native_gate_qualified': False, 'provider_calls_performed': False,
        'production_hook_deployed': False, 'canonical_tasks_accepted': 0,
        'acceptance_authority': False, 'completion_authority': False}
    write_new(ATTEMPT / 'capsule-preparation.json', wire(outcome), mode=0o600)
    print(json.dumps(outcome, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
