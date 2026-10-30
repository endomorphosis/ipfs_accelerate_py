"""Freeze an exact outer original-owner request after root freezes the packet.

Source reads/writes only: no Git, network, subprocess, imports of product code,
admission, capacity changes, or traversal of historical request input closures.
"""
from pathlib import Path
import hashlib
import json
import os
import stat

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
REQUEST = ROOT / 'publication-request-r01.json'
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-settlement-publication-20261017-r01')
REPO = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016/report_contract_proposal/daemon-settlement-qualification-v01'
BASE = '89dc25dbb32569bd668940af74c611221aa31148'


def need(value, reason):
    if not value: raise RuntimeError(reason)


def read(path, maximum=16 * 1024**2):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased_request_source_required')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and not before.st_mode & 2 and 0 <= before.st_size <= maximum, 'bounded_request_source_required')
        raw = stream.read(maximum + 1); after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'request_source_changed')
    return raw


def digest(path):
    # Executed gh is larger than the documentation reader's 16 MiB bound.
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased_execution_input_required')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and not before.st_mode & 2 and 0 <= before.st_size <= 64 * 1024**2, 'bounded_execution_input_required')
        value = hashlib.sha256(); size = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(block); need(size <= before.st_size, 'execution_input_grew'); value.update(block)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(size == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'execution_input_changed')
    return value.hexdigest()


def main():
    need(not REQUEST.exists() and not ATTEMPT.exists(), 'fresh_publication_request_and_attempt_required')
    selection = json.loads(read(HERE / 'SELECTION.json'))
    need(selection['status'] == 'root_finalized', 'explicit_root_final_docs_required')
    packet_path = HERE / 'PACKET.json'; packet = json.loads(read(packet_path))
    need(packet['schema'] == 'legal-ir-settlement-documentation-packet@1' and packet['base_head'] == BASE
         and packet['prefix'] == PREFIX and packet['github_qualification'] == 'skipped_by_user', 'exact_final_packet_required')
    manifest_path = HERE / 'SOURCE_MANIFEST.json'; manifest = json.loads(read(manifest_path))
    need(digest(manifest_path) == packet['publisher_source_manifest_sha256'], 'packet_publisher_manifest_changed')
    contract = json.loads(read(HERE / 'CONTRACT.json'))
    external = [{'path': str(REPO / PREFIX), 'bytes': 16 * 1024**2}, {'path': str(REPO / '.git'), 'bytes': 64 * 1024**2}]
    need(contract['storage_bytes'] == 256 * 1024**2 and contract['duration_seconds'] == 1800
         and contract['external_charges'] == external, 'exact_256MiB_and_80MiB_external_contract_required')
    pins = {str(packet_path): digest(packet_path), str(manifest_path): digest(manifest_path)}
    for row in manifest['files']:
        path = HERE / row['path']; raw = read(path)
        need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'publisher_source_row_changed')
        pins[str(path)] = row['sha256']
    for spelling, expected in manifest['input_pins'].items():
        path = Path(spelling)
        need('.git' not in path.parts and digest(path) == expected, 'immutable_executed_input_pin_required')
        pins[spelling] = expected
    for row in packet['files']:
        path = ROOT / row['path']; raw = read(path)
        need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'inert_packet_payload_changed')
        pins[str(path)] = row['sha256']
    # Direct existing documentation inputs only. Do not interpret or expand any
    # historical manifest/request JSON contained inside these inert documents.
    inventory = json.loads(read(HERE / 'EXISTING_REPORT_INVENTORY.json'))
    for row in inventory['files']:
        path = REPO / row['path']
        need(digest(path) == row['sha256'], 'preexisting_document_input_changed')
        pins[str(path)] = row['sha256']
    required = {str(HERE / 'publish_docs.py'), str(HERE / 'parent_attachment_gate.py'),
        '/usr/bin/git', '/usr/bin/gh', '/usr/bin/python3.12',
        '/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/run_image_owner_v05.py',
        str(REPO / 'ipfs_accelerate_py/agent_supervisor/runtime/ordinary_provider_cleanup.py'),
        str(REPO / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'),
        str(REPO / 'docs/architecture/legal_ir_supervisor_start_20261016/tasks.todo.md')}
    need(required <= set(pins) and len(pins) <= 1024, 'closed_bounded_actual_publication_input_closure_required')
    need(not any('.git' in Path(spelling).parts for spelling in pins), 'mutable_git_metadata_pins_refused')
    command = ['/usr/bin/python3.12', '-B', str(HERE / 'publish_docs.py'), digest(packet_path), str(REQUEST)]
    request = {'schema': 'legal-ir-native-orchestration-request@1', 'name': 'legal-ir-settlement-publication-20261017-r01',
        'purpose': 'controller_preparation', 'duration_seconds': 1800, 'attempt_directory': str(ATTEMPT),
        'storage_bytes': 256 * 1024**2, 'external_charges': external, 'command': command, 'cwd': str(HERE),
        'source_sha256': pins, 'declared_source_mutations': [
            'Exact additive documentation generation on original owned draft PR292 branch from fixed89dc25d base only; preserve all506 prior report documents.',
            'Original V05 parent owns actual native admission/group cleanup and fresh global storage checks; exact256MiB budget includes80MiB external charges. No quota or other-owner changes.',
            'Historical request/manifest JSON files are inert documentation; only explicitly listed source inputs are read, no recursive historical request closures are executed.',
            'No runtime installation, canonical board/state/cap/counter mutation, providers/models, CI/GitHub qualification or PR merge.']}
    raw = (json.dumps(request, sort_keys=True, indent=2) + '\n').encode()
    need(len(raw) <= 1024**2, 'bounded_outer_request_required')
    fd = os.open(REQUEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'request': str(REQUEST), 'sha256': hashlib.sha256(raw).hexdigest(),
        'source_pins': len(pins), 'storage_bytes': request['storage_bytes'],
        'external_bytes': sum(row['bytes'] for row in external)}, sort_keys=True))


if __name__ == '__main__': main()
