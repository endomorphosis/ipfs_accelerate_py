"""Close only the reviewed completed inference-publication deployment generation.

Preparation is inert. Run with --close only after final receipts, cleanup,
README-final.md and package-ready.json have been supplied. This script never
starts Docker, executes task/model code, or changes production owners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

BASE = Path(__file__).parent
A = Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
OUT = A / 'docs/agent_supervisor/evidence/source384-docker-inference-publication-20261003'

HOST_REQUIRED = (
    'archive-audit.json', 'before-pins.json', 'bundle-command.json',
    'bundle-exit.json', 'bundle.stdout', 'bundle.stderr',
    'controls-command.json', 'controls-exit.json', 'controls-source-pins.json',
    'controls.log', 'controls.xml', 'deployment-wheel.patch',
    'frozen-d-pins.json', 'prepared-a-pins.json', 'pandas-compatibility.json',
    'wheel-provenance.json', 'wheel-transport-inventory.json', 'publication-controls-inventory.json',
    'plan.json', 'run-admission.json', 'run-command.json', 'run_qualification.py',
    'run.log', 'exit.json', 'task-public-inputs.json', 'cleanup-observation.json',
)
HOST_OPTIONAL = (
    'result.json', 'failure.json', 'docker-enforced-limits.json',
    'independent-review.json', 'independent-archive-review.json', 'independent-bundle-audit.json', 'independent-success-content-audit.json',
)
DOCKER_OPTIONAL = (
    'admission-estimate.json', 'resources.json', 'resource-probe.stdout',
    'resource-probe.stderr', 'qualification.json', 'source384-probe.stdout',
    'source384-probe.stderr', 'source384-probe-status.json',
    'source384-result.json', 'source384-context.json',
)
DEPLOYMENT_LOGS = (
    'original-inputs', 'root-create', 'archive-extract', 'source-generation',
    'system-install', 'python-runtime-install', 'python-create',
    'torch-cpu-wheel-verify', 'torch-cpu-install', 'python-install', 'account-create', 'native-extensions',
    'native-imports', 'empty-native-start', 'retained-inputs',
)
TEST_SOURCES = (
    'benchmarks/agent_supervisor/container_coding/terminal_deployment.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_deployment.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_source384_transport.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_torch_wheel.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_source384_qualification.py',
)
EXTRA_ARCHIVE_OWNERS = (
    'datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_384.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/codebase_model_generation.py',
    'datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_schema_lake.py',
    'source/ipfs_accelerate_py/agent_supervisor/runtime/source384_config.py',
    'source/ipfs_accelerate_py/agent_supervisor/runtime/security_autoencoder_advisor.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/cache.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_units_384_worker.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/duckdb_ingest.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/python_frontend.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/semantic_index/scanner.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/semantic_index/snapshot.py',
    'datasets/ipfs_datasets_py/logic/software_contracts/semantic_index/python_analysis.py',
)


def read(name):
    return json.loads((BASE / name).read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--close', action='store_true', required=True)
    parser.parse_args()
    ready = read('package-ready.json')
    summary = read('qualification-summary.json')
    cleanup = read('cleanup-observation.json')
    assert ready['final_result_reviewed'] is True
    assert cleanup['container_absent'] is True
    assert ready['readme_sha256'] == digest(BASE / 'README-final.md')
    assert ready['summary_sha256'] == digest(BASE / 'qualification-summary.json')
    assert summary['provider_calls'] == 0 and summary['benchmark_result'] is False
    assert summary['official_verifier_executed'] is False
    assert not OUT.exists(), 'closed evidence destination must be fresh'
    assert (BASE / 'result.json').is_file() != (BASE / 'failure.json').is_file()
    archive = read('bundle/manifest.json')
    assert digest(BASE / 'bundle/runtime.tar.gz') == archive['archive_sha256']
    inventory = {row['path']: row for row in archive['files']}
    assert len(inventory) == len(archive['files'])
    pins = read('prepared-a-pins.json') + read('frozen-d-pins.json')
    for row in pins:
        assert inventory[row['path']]['sha256'] == row['sha256']
    selected = {}

    def select(relative, destination=None, required=True):
        path = BASE / relative
        if not path.exists() and not required:
            return
        assert path.is_file() and not path.is_symlink(), relative
        assert path.stat().st_size <= 40 * 1024 * 1024, relative
        target = destination or ('host/' + relative)
        assert target not in selected, target
        selected[target] = path

    for name in HOST_REQUIRED:
        select(name)
    for name in HOST_OPTIONAL:
        select(name, required=False)
    select('qualification.json', 'controls/qualification.json')
    select('bundle/manifest.json', 'bundle/manifest.json')
    select('README-final.md', 'README.md')
    select('qualification-summary.json', 'qualification.json')
    select('package-ready.json', 'host/package-ready.json')
    select('package_evidence.py', 'host/package_evidence.py')
    for name in DOCKER_OPTIONAL:
        select('docker-01/' + name, 'docker-01/' + name, required=False)
    for name in ('admission-estimate.json', 'resources.json',
                 'resource-probe.stdout', 'resource-probe.stderr'):
        relative = 'docker-01/final-resources/' + name
        select(relative, relative, required=False)
    for name in DEPLOYMENT_LOGS:
        relative = 'docker-01/deployment/' + name + '.log'
        select(relative, relative, required=False)
    select('docker-01/deployment/deployment.json',
           'docker-01/deployment/deployment.json', required=False)
    for name in TEST_SOURCES:
        select('final-sources/' + name, 'final-sources/' + name)
    for name in TEST_SOURCES[:2]:
        select('before/' + name, 'before/' + name)
    for row in read('publication-controls-inventory.json')['files']:
        relative = row['path']
        assert relative.startswith('publication-controls/') and '..' not in Path(relative).parts
        assert digest(BASE / relative) == row['sha256']
        select(relative, relative)
    for row in read('wheel-transport-inventory.json')['files']:
        relative = row['path']
        assert relative.startswith('wheel-transport/') and '..' not in Path(relative).parts
        assert digest(BASE / relative) == row['sha256']
        select(relative, relative)
    for row in read('wheel-provenance.json')['files']:
        relative = row['path']
        assert relative.startswith('wheel-provenance/') and '..' not in Path(relative).parts
        assert (BASE / relative).suffix in {'.json', '.log', '.py'}
        assert digest(BASE / relative) == row['sha256']
        select(relative, relative)
    native = BASE / 'docker-01/native-inference.json'
    if native.exists():
        assert ready['native_inference_retained_sha256'] == digest(native)
        if ready['native_inference_export'] == 'exclude_source_payload':
            audit = read('native-inference-public-audit.json')
            assert audit['original_export']['sha256'] == digest(native)
            assert audit['disposition'] == 'retain_original_locally_exclude_from_public_package'
            select('audit_native_export.py')
            select('native-inference-public-audit.json')
            select('native-inference-verification.json')
        else:
            assert ready['native_inference_export'] == 'include_reviewed_public_export'
            assert ready['native_inference_public_sha256'] == digest(native)
            select('docker-01/native-inference.json', 'docker-01/native-inference.json')
    owners = set(EXTRA_ARCHIVE_OWNERS) | {row['path'] for row in pins}
    assert owners <= inventory.keys()
    OUT.mkdir(parents=True)
    for name, path in sorted(selected.items()):
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    remaining = set(owners)
    with tarfile.open(BASE / 'bundle/runtime.tar.gz', 'r|gz') as stream:
        for member in stream:
            if member.name not in remaining:
                continue
            assert member.isfile() and member.size == inventory[member.name]['bytes']
            assert member.size <= 4 * 1024 * 1024
            raw = stream.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == inventory[member.name]['sha256']
            target = OUT / 'archive-owners' / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            remaining.remove(member.name)
    assert not remaining
    rows = [dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size,
                 sha256=digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
    manifest = dict(schema='closed-source384-docker-inference-publication-evidence@1', files=rows)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    result = dict(path=str(OUT), files=len(rows), bytes=sum(row['bytes'] for row in rows),
                  manifest_sha256=digest(OUT / 'manifest.json'))
    (BASE / 'package-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
