"""Build and independently compare the reviewed production cache bundle.

This script must only run after applying and qualifying the selected owners.
It changes no previous archive, checkpoint, task, or admission policy.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

BASE = Path(__file__).resolve().parent
WORK = Path('/home/barberb/lift_coding')
A = WORK / '.worktrees/ir-release-accelerate-20261002'
D = WORK / '.worktrees/ir-release-datasets-20261002'
PREVIOUS = WORK / 'artifacts/source384-production-cache-20261003'
POLICY = 'source384-native-aarch64-dontneed@1'
EXPECTED_HEADS = {
    'accelerate': '907dae9bc05affedfe5f97e56da3197d4f110cb9',
    'datasets': '8879dce383b680de7d1643295af400b0f617eff0',
}
PRIOR_MANIFEST_SHA256 = 'eb4913c3a11dbf93d166f0f333d80f314a2ce876f9673763b27f50a1f33d67f0'
PREFIX = 'source/benchmarks/agent_supervisor/container_coding/'
CHANGED = {
    PREFIX + 'terminal_initial_context.py',
    PREFIX + 'terminal_container_supervisor.py',
    PREFIX + 'terminal_source384_qualification.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py',
    'datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_embedding_runtime.py',
    'datasets/ipfs_datasets_py/logic/formalization/autoencoder/long_span_source_value_training.py',
}
ADDED = {
    PREFIX + 'terminal_resource_diagnostics.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/codebase_header_context.py',
    'datasets/ipfs_datasets_py/logic/formalization/autoencoder/action_contrastive_decoder_training.py',
    'datasets/ipfs_datasets_py/logic/formalization/autoencoder/action_factorized_clause_decoder_experiment.py',
}
# Root is qualifying an additional failure-diagnostics helper; add its exact
# source names and final producer SHA before committing these execution pins.
EXPECTED_DRIVER_SHA256 = '688acea0a9017a901fd26cae42ae63f0d198418b9d0e27c1b22c6697e1caef19'


def owner(name):
    prefix, relative = name.split('/', 1)
    return {'source': A, 'datasets': D}[prefix] / relative


def write(name, value):
    (BASE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()


def main():
    if any(value is None for value in EXPECTED_HEADS.values()) or EXPECTED_DRIVER_SHA256 is None:
        raise RuntimeError('final applied and tested source commit pins are pending')
    if (BASE / 'bundle').exists() or (BASE / 'build-command.json').exists():
        raise RuntimeError('fresh output required; retain prior attempts')
    prior_path = PREVIOUS / 'bundle/manifest.json'
    assert sha(prior_path) == PRIOR_MANIFEST_SHA256
    prior = json.loads(prior_path.read_text())
    assert prior['archive_sha256'] == 'a36bf3550eb17a0c3a1f190f5cad2364a66b07994e9ecacd6637666d5df761e0'
    recipe = json.loads((PREVIOUS / 'build-command.json').read_text())
    environment_recipe_path = WORK / 'artifacts/source384-full-supervisor-preflight-20261003/prepare-command.json'
    environment_recipe = json.loads(environment_recipe_path.read_text())
    build_env = environment_recipe['env']
    argv = recipe['argv'][:]
    argv[0] = str(WORK / '.venvs/terminal-bench-harbor/bin/python')
    argv[argv.index('--output') + 1] = str(BASE / 'bundle')
    assert argv.count('--setup-cache-policy') == 1 and argv[argv.index('--setup-cache-policy') + 1] == POLICY
    heads = {name: subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=path, text=True).strip() for name, path in [('accelerate', A), ('datasets', D)]}
    assert heads == EXPECTED_HEADS, heads
    inherited_pins = json.loads((PREVIOUS / 'frozen-production-pins.json').read_text())
    assert all(sha(owner(name)) == pin for name, pin in inherited_pins.items() if name not in CHANGED)
    frozen = {name: sha(owner(name)) for name in sorted(set(inherited_pins) | CHANGED | ADDED)}
    assert frozen[PREFIX + 'terminal_container_supervisor.py'] == EXPECTED_DRIVER_SHA256
    disk = shutil.disk_usage(BASE)
    required = 2 * sum(row['bytes'] for row in prior['files'])
    write('resources-before.json', dict(at=datetime.now(timezone.utc).isoformat(),
        free_bytes=disk.free, conservative_workspace_bytes=required,
        scope='bundle creation only, no native resource admission'))
    if disk.free < required:
        raise RuntimeError('insufficient free disk for retained fresh bundle')
    write('frozen-production-pins.json', frozen)
    write('build-command.json', dict(argv=argv, cwd=str(A), env=build_env,
        environment_recipe=str(environment_recipe_path), environment_recipe_sha256=sha(environment_recipe_path),
        source_revisions=heads, builder_sha256=sha(Path(__file__)),
        preceding_manifest_sha256=PRIOR_MANIFEST_SHA256,
        preceding_archive_sha256=prior['archive_sha256'],
        preceding_build_audit_sha256=sha(PREVIOUS / 'build-audit.json'),
        inherited_producer_pins_sha256=sha(PREVIOUS / 'frozen-production-pins.json'),
        expected_changed=sorted(CHANGED), expected_added=sorted(ADDED),
        provider_calls=0, benchmark_executed=False))
    started = time.monotonic()
    with (BASE / 'build.stdout').open('w') as out, (BASE / 'build.stderr').open('w') as err:
        result = subprocess.run(argv, cwd=A, env=dict(os.environ, **build_env),
            stdout=out, stderr=err)
    write('build-exit.json', dict(returncode=result.returncode, seconds=time.monotonic() - started))
    if result.returncode:
        raise SystemExit(result.returncode)
    manifest_path = BASE / 'bundle/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    mutable_metadata = {'files', 'archive_sha256', 'setup_cache'}
    assert {key: value for key, value in manifest.items() if key not in mutable_metadata} == {
        key: value for key, value in prior.items() if key not in mutable_metadata}
    old_rows = {r['path']: r for r in prior['files']}
    rows = {r['path']: r for r in manifest['files']}
    assert len(rows) == len(manifest['files'])
    changed = {name for name in old_rows.keys() & rows.keys() if old_rows[name] != rows[name]}
    added = rows.keys() - old_rows.keys()
    removed = old_rows.keys() - rows.keys()
    assert changed == CHANGED, changed
    assert added == ADDED, added
    assert not removed, removed
    for name, pin in frozen.items():
        assert rows[name]['sha256'] == pin == sha(owner(name))
    for key in ('source384', 'torch_cpu_wheel'):
        assert manifest[key] == prior[key], key
    assert manifest['setup_cache']['policy'] == POLICY
    archive = BASE / 'bundle/runtime.tar.gz'
    assert sha(archive) == manifest['archive_sha256']
    verified = set()
    total = 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            assert member.isfile() and member.name not in verified
            row = rows[member.name]
            assert member.size == row['bytes'] and member.mode == row['mode']
            assert member.uid == 0 and member.gid == 0
            h = hashlib.sha256()
            body = stream.extractfile(member)
            for raw in iter(lambda: body.read(1024 * 1024), b''):
                h.update(raw)
            assert h.hexdigest() == row['sha256'], member.name
            verified.add(member.name)
            total += member.size
    assert verified == set(rows)
    selection = json.loads((BASE / 'bundle/setup-cache-selection.json').read_text())
    assert selection['manifest_sha256'] == sha(manifest_path)
    after_heads = {name: subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=path, text=True).strip() for name, path in [('accelerate', A), ('datasets', D)]}
    assert after_heads == heads
    write('build-audit.json', dict(archive_sha256=manifest['archive_sha256'],
        manifest_sha256=sha(manifest_path), verified_members=len(verified),
        logical_bytes=total, compressed_bytes=archive.stat().st_size,
        changed=sorted(changed), added=sorted(added), removed=sorted(removed),
        source384_unchanged=True, torch_wheel_unchanged=True,
        nonmutable_manifest_metadata_unchanged=True,
        source_revisions=heads, source_revisions_unchanged=True,
        builder_sha256=sha(Path(__file__)), preceding_manifest_sha256=PRIOR_MANIFEST_SHA256,
        tar_hash_mode_ownership_checked=True, setup_cache_selection=selection,
        original_task_read=False, provider_calls=0, benchmark_executed=False))
    print(json.dumps(json.loads((BASE / 'build-audit.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
