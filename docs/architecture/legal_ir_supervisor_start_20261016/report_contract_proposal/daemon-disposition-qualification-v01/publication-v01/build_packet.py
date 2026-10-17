"""Source-authoring freezer for an explicitly selected documentation packet.

No Git, subprocess, network, product import or native authority. Root finalizes
SELECTION.json only after its observations and readiness/tasks are complete.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
BASE = 'a1b1f32826e58913837308330d8ec6cc8589222a'
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016/report_contract_proposal/daemon-disposition-qualification-v01'


def need(value, reason):
    if not value: raise RuntimeError(reason)


def safe(value):
    path = PurePosixPath(value)
    need(type(value) is str and value and not path.is_absolute() and path.as_posix() == value
         and not any(p in {'', '.', '..', '.git', '__pycache__'} for p in path.parts)
         and not any(c in value for c in '\x00\r\n\t'), 'canonical_selected_relative_path_required')
    return ROOT / path


def read(path):
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'selected_alias_refused')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        metadata = os.fstat(stream.fileno())
        need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
             and metadata.st_uid in {0, os.getuid()} and not metadata.st_mode & 2
             and 0 <= metadata.st_size <= 4 * 1024**2, 'selected_bounded_owned_regular_file_required')
        raw = stream.read(4 * 1024**2 + 1); after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid,
                         m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == metadata.st_size and witness(metadata) == witness(after) == witness(path.lstat()),
         'selected_source_changed_during_read')
    return raw


def main():
    selection = json.loads(read(HERE / 'SELECTION.json'))
    need(selection['schema'] == 'legal-ir-disposition-publication-selection@1'
         and selection['status'] == 'root_finalized', 'root_must_finalize_explicit_selection_first')
    names = set()
    for directory in selection['directories']:
        path = safe(directory)
        need(path.is_dir() and not path.is_symlink() and directory != 'publication-v01',
             'explicit_nonrecursive_source_bundle_directory_required')
        for base, directories, files in os.walk(path, followlinks=False):
            need(not any((Path(base) / name).is_symlink() or name in {'.git', '__pycache__'} for name in directories),
                 'selected_directory_alias_or_cache_refused')
            for name in files:
                item = Path(base) / name
                need(not name.endswith(('.pyc', '.tmp')) and not name.startswith('.'), 'source_bundle_temporary_file_refused')
                names.add(item.relative_to(ROOT).as_posix())
    for name in selection['files']:
        safe(name); names.add(name)
    need({'READINESS.md', 'TASKS.json'} <= names,
         'root_readiness_and_tasks_required')
    need('publication-v01/PACKET.json' not in names
         and not any(name.startswith('publication-root-observation') or name.startswith('publication-request') for name in names),
         'future_publication_input_cycle_refused')
    need(0 < len(names) <= 256, 'bounded_explicit_publication_count_required')
    observations = selection['root_observations']
    need(type(observations) is list and observations and set(observations) <= names,
         'explicit_finalized_root_observations_required')
    for name in observations:
        observation = json.loads(read(safe(name)))
        need(observation.get('native_finalized') is True and observation.get('acceptance_authority') is False,
             'root_observation_must_record_actual_native_finalization_and_no_acceptance')
    rows = []
    for name in sorted(names):
        raw = read(safe(name))
        rows.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    need(sum(row['bytes'] for row in rows) <= 8 * 1024**2, 'bounded_selected_publication_bytes_required')
    packet = {'schema': 'legal-ir-disposition-documentation-packet@1', 'base_head': BASE, 'branch': BRANCH,
        'prefix': PREFIX, 'files': rows, 'github_qualification': 'skipped_by_user', 'acceptance_authority': False,
        'runtime_installed': False, 'root_observations': observations,
        'publisher_source_manifest_sha256': hashlib.sha256(read(HERE / 'SOURCE_MANIFEST.json')).hexdigest()}
    target = HERE / 'PACKET.json'
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'w') as stream:
        json.dump(packet, stream, sort_keys=True, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'packet': str(target), 'sha256': hashlib.sha256(read(target)).hexdigest(),
        'files': len(rows), 'bytes': sum(row['bytes'] for row in rows)}, sort_keys=True))


if __name__ == '__main__': main()
