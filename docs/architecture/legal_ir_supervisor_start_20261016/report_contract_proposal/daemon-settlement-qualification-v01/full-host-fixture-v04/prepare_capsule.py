"""Copy exact complete capsule under the caller's genuine V05 native owner.

Unexecuted source. This starts no product module, Git, native scheduler or model.
Its JSON/source inventory is provenance, never a native lease or runtime grant.
"""
import importlib.util
import os
from pathlib import Path
import sys

from fixture_io import canonical, fsync_dir, need, read, sha, source_guard, strict, wire, write_new

HERE = Path(__file__).absolute().parent
ASSEMBLER = HERE.parent / 'daemon-integration-v03/build_patch.py'
ASSEMBLER_SHA = 'f5199a384dfc8ab642582bb68fe78765f61b6958471597180aab4f25c270006e'
GATE = HERE.parent / 'repository-fixture-v02/parent_attachment_gate.py'
GATE_SHA = 'dba5662443619358e2225d0cf7ccaa2e2b6c5bf51b0a60d2d1809004d1667fb3'
PARENT_NAME = 'legal-ir-full-daemon-capsule-20261017-r02'


def main():
    need(len(sys.argv) == 5, 'reviewed_capsule_spec_SHA_source_manifest_SHA_and_parent_request_required')
    source_guard(HERE, sys.argv[3])
    root = canonical(os.environ['TMPDIR'])
    need(root.is_dir() and root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o077 == 0,
         'actual_private_preparation_attempt_required')
    read(GATE, GATE_SHA)
    gate_loader = importlib.util.spec_from_file_location('exact_original_capsule_preparation_attachment_gate', GATE)
    need(gate_loader is not None and gate_loader.loader is not None, 'exact_preparation_gate_loader_required')
    gate = importlib.util.module_from_spec(gate_loader)
    gate_loader.loader.exec_module(gate)
    gate.wait_for_original_generic_parent(sys.argv[4], attempt_directory=root,
        expected_request_name=PARENT_NAME, storage_bytes=134217728,
        worker_command=[sys.executable, '-B', str(Path(__file__).absolute()), *sys.argv[1:]], timeout_seconds=60)
    spec = strict(read(sys.argv[1], sys.argv[2], maximum=1024**2))
    need(spec.get('target_capsule_root') == str(root / 'capsule'), 'original_preparation_attempt_capsule_required')
    daemon_rows = [row for row in spec['overrides'] if row['path'].endswith('/todo_daemon/implementation_daemon.py')]
    need(len(daemon_rows) == 1 and daemon_rows[0]['source_path'] == str(root / 'assembled-daemon.py'),
         'fresh_charged_full_daemon_assembly_output_required')
    read(ASSEMBLER, ASSEMBLER_SHA)
    assembler_loader = importlib.util.spec_from_file_location('exact_frozen_V03_daemon_source_assembler', ASSEMBLER)
    need(assembler_loader is not None and assembler_loader.loader is not None, 'exact_daemon_source_assembler_loader_required')
    assembler = importlib.util.module_from_spec(assembler_loader)
    assembler_loader.loader.exec_module(assembler)  # Source authoring only; main remains uncalled.
    original, base, proposed = assembler.assemble()
    daemon_raw = proposed.encode()
    need(sha(daemon_raw) == daemon_rows[0]['sha256'] and len(daemon_raw) == daemon_rows[0]['bytes'],
         'actual_assembled_full_daemon_bytes_differ')
    write_new(root / 'assembled-daemon.py', daemon_raw, mode=0o444)
    loader = importlib.util.spec_from_file_location('frozen_full_daemon_capsule_inventory_builder', HERE / 'build_capsule_inventory.py')
    need(loader is not None and loader.loader is not None, 'exact_engineering_builder_loader_required')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)  # Standard-library source only, already pinned.
    inventory = module.build_inventory(spec, sys.argv[2])
    capsule = canonical(inventory['target_capsule_root'])
    need(not os.path.lexists(capsule), 'fresh_complete_capsule_required')
    inventory_raw = wire(inventory)
    write_new(root / 'capsule-source-inventory.json', inventory_raw)
    capsule.mkdir(mode=0o700)
    rows = []
    for row in inventory['files']:
        destination = canonical(capsule / row['path'])
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        raw = read(row['source_path'], row['sha256'])
        need(len(raw) == row['bytes'], 'original_source_bytes_changed')
        write_new(destination, raw, mode=0o444)
        rows.append({key: row[key] for key in ('path', 'bytes', 'sha256')})
    read(sys.argv[1], sys.argv[2], maximum=1024**2)
    for row in inventory['files']:
        need(len(read(row['source_path'], row['sha256'])) == row['bytes'], 'source_changed_after_copy')
    for spelling, digest in inventory['freeze_input_pins'].items():
        read(spelling, digest)
    manifest = {'schema': 'legal-ir-full-host-capsule@1', 'capsule_root': str(capsule),
        'base_manifest_sha256': inventory['base_manifest_sha256'], 'source_inventory_sha256': sha(inventory_raw),
        'reviewed_spec_sha256': sys.argv[2], 'reviewed_composed_daemon_sha256': inventory['reviewed_composed_daemon_sha256'],
        'payload_file_count': len(rows), 'payload_bytes': sum(row['bytes'] for row in rows), 'files': rows,
        'runtime_enabled': False, 'provider_calls_performed': False, 'full_host_qualified': False,
        'acceptance_authority': False, 'completion_authority': False}
    manifest_raw = wire(manifest)
    write_new(capsule / 'SOURCE_MANIFEST.json', manifest_raw, mode=0o444)
    directories = [Path(path) for path, _dirs, _files in os.walk(capsule, followlinks=False)]
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        canonical(directory); fsync_dir(directory); directory.chmod(0o500)
    fsync_dir(root)
    source_guard(HERE, sys.argv[3])
    result = {'schema': 'legal-ir-full-host-capsule-preparation@1', 'passed': True,
        'capsule_root': str(capsule), 'capsule_manifest_sha256': sha(manifest_raw),
        'capsule_inventory_sha256': sha(inventory_raw), 'source_manifest_sha256': sys.argv[3],
        'reviewed_spec_sha256': sys.argv[2], 'payload_file_count': len(rows),
        'payload_bytes': manifest['payload_bytes'], 'product_imported': False,
        'provider_calls_performed': False, 'Git_calls_performed': False,
        'native_parent_finalization_claimed': False, 'full_host_qualified': False,
        'canonical_tasks_accepted': 0, 'acceptance_authority': False, 'completion_authority': False}
    write_new(root / 'capsule-preparation.json', wire(result))
    print(wire(result).decode(), end='', flush=True)


if __name__ == '__main__':
    main()
