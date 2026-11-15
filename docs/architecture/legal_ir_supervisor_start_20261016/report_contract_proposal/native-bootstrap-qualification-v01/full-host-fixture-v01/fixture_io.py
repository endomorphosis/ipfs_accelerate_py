"""Bounded standard-library fixture I/O; no product imports or admission."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    path = Path(value)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical_fixture_path_required')
    return path


def read(path, expected=None, maximum=16 * 1024**2):
    path = canonical(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid in {0, os.getuid()} and before.st_nlink == 1
             and before.st_mode & 2 == 0 and 0 <= before.st_size <= maximum, 'bounded_owned_fixture_file_required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_mode, m.st_uid, m.st_gid,
                         m.st_nlink, m.st_size, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'fixture_file_changed')
    if expected is not None:
        need(type(expected) is str and len(expected) == 64 and sha(raw) == expected, 'fixture_file_SHA_changed')
    return raw


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_fixture_JSON_field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_fixture_JSON')))


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def fsync_dir(path):
    fd = os.open(canonical(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, raw, mode=0o600):
    path = canonical(path)
    need(type(raw) is bytes and len(raw) <= 4 * 1024**2 and path.parent.is_dir(), 'bounded_fresh_output_required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fsync_dir(path.parent)


def source_guard(here, expected):
    raw = read(here / 'SOURCE_MANIFEST.json', expected, maximum=1024**2)
    manifest = strict(raw)
    need(manifest.get('schema') == 'legal-ir-native-bootstrap-full-host-fixture-source@1', 'exact_fixture_source_schema_required')
    guards = dict(manifest['input_pins'])
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == row['path'],
             'canonical_fixture_source_row_required')
        data = read(here / row['path'], row['sha256'])
        need(len(data) == row['bytes'], 'fixture_source_size_changed')
        guards[str(here / row['path'])] = row['sha256']
    for spelling, digest in guards.items():
        read(spelling, digest, maximum=256 * 1024**2)
    return manifest
