"""Unexecuted preparation source; run only after separate native admission.

This builds a complete source inventory from the immutable original capsule
and exactly four separately reviewed overrides. It imports no product module,
runs no assembler/Git/provider, copies no payload and grants no runtime power.
The caller must first materialize the reviewed complete daemon source under
admission, then provide its exact assembly/source digest in the frozen spec.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

RUNS = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
BASE = RUNS / 'legal-ir-full-daemon-capsule-20261017-r02/capsule'
BASE_SHA = '41a0fdbf91a4aff6240db96483afc85a3b2e25d73aad6c7a5aa0884014e1eb0b'
TODO = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/'
OVERRIDES = {TODO + name for name in ('implementation_daemon.py',
    'report_only_native_host.py', 'attempt_disposition.py', 'report_only_disposition_gate.py')}
MAX_SOURCE, MAX_PAYLOAD = 16 * 1024**2, 128 * 1024**2


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    need(type(value) is str, 'explicit_absolute_path_required')
    path = Path(value)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical_unaliased_path_required')
    return path


def stable(path, *, expected=None, size=None, maximum=MAX_SOURCE):
    path = canonical(str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 2 == 0 and 0 <= before.st_size <= maximum, 'bounded_owned_source_required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_mode, m.st_uid, m.st_gid,
                         m.st_nlink, m.st_size, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'source_changed')
    if expected is not None:
        need(type(expected) is str and len(expected) == 64 and digest(raw) == expected, 'source_digest_differs')
    if size is not None:
        need(type(size) is int and size == len(raw), 'source_bytes_differ')
    return raw


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_JSON_field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_JSON')))


def relative(value):
    need(type(value) is str, 'explicit_relative_source_required')
    path = PurePosixPath(value)
    need(not path.is_absolute() and path.as_posix() == value and '..' not in path.parts
         and path.parts[0] == 'ipfs_accelerate_py' and path.suffix == '.py'
         and '.git' not in path.parts and '__pycache__' not in path.parts, 'canonical_python_source_required')
    return value


def build_inventory(spec, spec_sha):
    need(type(spec) is dict and set(spec) == {'schema', 'target_capsule_root', 'assembly_path',
        'assembly_sha256', 'freeze_input_pins', 'overrides'}
        and spec['schema'] == 'legal-ir-full-host-capsule-reviewed-source-spec@1', 'closed_reviewed_spec_required')
    target = canonical(spec['target_capsule_root'])
    need(RUNS in target.parents and target.name == 'capsule' and not os.path.lexists(target)
         and target != BASE, 'fresh_owned_capsule_destination_required')
    freeze_pins = spec['freeze_input_pins']
    need(type(freeze_pins) is dict and 2 <= len(freeze_pins) <= 64, 'bounded_reviewed_input_closure_required')
    for spelling, sha in freeze_pins.items():
        stable(spelling, expected=sha)
    assembly_path = canonical(spec['assembly_path'])
    need(freeze_pins.get(str(assembly_path)) == spec['assembly_sha256'], 'reviewed_assembly_in_input_closure_required')
    assembly = strict(stable(assembly_path, expected=spec['assembly_sha256']))
    need(assembly.get('schema') in {'legal-ir-disposition-entry-source-assembly@1'}
         and assembly.get('runtime_enabled') is False and type(assembly.get('proposed_sha256')) is str,
         'actual_frozen_disabled_daemon_assembly_required')
    raw_base = stable(BASE / 'SOURCE_MANIFEST.json', expected=BASE_SHA, maximum=4 * 1024**2)
    base = strict(raw_base)
    need(base.get('schema') == 'legal-ir-full-host-capsule@1'
         and base.get('capsule_root') == str(BASE) and base.get('payload_file_count') == 2810
         and type(base.get('files')) is list and len(base['files']) == 2810, 'exact_original_capsule_required')
    original = {}
    for row in base['files']:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed_original_source_row_required')
        name = relative(row['path'])
        need(name not in original, 'duplicate_original_source_row')
        stable(BASE / name, expected=row['sha256'], size=row['bytes'])
        original[name] = row
    need(sum(row['bytes'] for row in original.values()) == base.get('payload_bytes') <= MAX_PAYLOAD,
         'original_complete_byte_total_required')
    overrides = spec['overrides']
    need(type(overrides) is list and len(overrides) == len(OVERRIDES), 'exact_four_overrides_required')
    selected = {}
    for row in overrides:
        need(type(row) is dict and set(row) == {'path', 'source_path', 'bytes', 'sha256'}, 'closed_override_source_row_required')
        name = relative(row['path'])
        need(name in OVERRIDES and name not in selected, 'exact_unique_override_required')
        source = canonical(row['source_path'])
        need(BASE not in source.parents and target not in source.parents, 'separate_reviewed_override_source_required')
        stable(source, expected=row['sha256'], size=row['bytes'])
        selected[name] = row
    need(set(selected) == OVERRIDES and selected[TODO + 'implementation_daemon.py']['sha256'] == assembly['proposed_sha256'],
         'exact_reviewed_full_daemon_override_required')
    rows = []
    for name in sorted(set(original) | OVERRIDES):
        if name in selected:
            row = selected[name]
            rows.append(dict(row, origin='reviewed_frozen_override'))
        else:
            rows.append(dict(original[name], source_path=str(BASE / name), origin='original_readonly_capsule'))
    need(len(rows) <= 4096 and sum(row['bytes'] for row in rows) <= MAX_PAYLOAD, 'bounded_complete_capsule_required')
    return {'schema': 'legal-ir-disposition-entry-capsule-source-inventory@1',
        'source_spec_sha256': spec_sha, 'base_capsule_root': str(BASE), 'base_manifest_sha256': BASE_SHA,
        'target_capsule_root': str(target), 'freeze_input_pins': freeze_pins,
        'reviewed_composed_daemon_sha256': assembly['proposed_sha256'], 'files': rows,
        'payload_file_count': len(rows), 'payload_bytes': sum(row['bytes'] for row in rows),
        'runtime_enabled': False, 'native_admission_granted': False, 'full_host_qualified': False,
        'provider_calls_performed': False, 'acceptance_authority': False, 'completion_authority': False}


def main():
    need(len(sys.argv) == 4, 'reviewed_spec_path_SHA_and_fresh_inventory_output_required')
    raw = stable(sys.argv[1], expected=sys.argv[2], maximum=1024**2)
    spec = strict(raw)
    inventory = build_inventory(spec, sys.argv[2])
    output = canonical(sys.argv[3])
    need(RUNS in output.parents and output.parent.is_dir() and not os.path.lexists(output),
         'fresh_charged_inventory_output_required')
    need(output.parent.stat().st_uid == os.getuid() and output.parent.stat().st_mode & 0o077 == 0,
         'private_inventory_parent_required')
    encoded = (json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(encoded) <= 4 * 1024**2, 'bounded_inventory_output_required')
    # Re-read the reviewed source closure immediately before the write.
    need(build_inventory(strict(stable(sys.argv[1], expected=sys.argv[2], maximum=1024**2)), sys.argv[2]) == inventory,
         'source_inventory_changed_before_write')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({'schema': 'legal-ir-full-host-capsule-source-inventory-authored@1',
        'output': str(output), 'inventory_sha256': digest(encoded), 'payload_file_count': inventory['payload_file_count'],
        'payload_bytes': inventory['payload_bytes'], 'native_finalization_claimed': False,
        'capsule_payload_copied': False, 'product_imported': False}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
