"""Retain a fresh, offline, source-pinned supervisor qualification run."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-context-gaps-20261004')
DATASETS = Path('/home/barberb/lift_coding/.worktrees/ir-supervisor-contracts-datasets-20261004')
OUT = Path(__file__).parent
OWNERS = [
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/production_daemon_route.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/production_provider_cli.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/production_context_slice.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/production_reviewed_effect.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/production_provider_attestation.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py',
    'ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py',
    'ipfs_accelerate_py/testing/pytest_ast_seal.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/security_source_program_advisor_384.py',
    'ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py',
    'ipfs_accelerate_py/llm_router.py',
]
ASSETS = [
    Path.home() / '.cache/huggingface/hub/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd/model.safetensors',
    Path.home() / '.cache/huggingface/hub/models--Publicus--intent-ir-autoencoder/blobs/66cc213ddf4a4c5292718df33db0821642ac6176',
    Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors',
    Path('/home/barberb/lift_coding/artifacts/distributed384-20261001/run-01/security_ir/coordinator/checkpoints/2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5.json'),
]


def digest(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return {'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', *arguments], cwd=path, text=True).strip()


def snapshot(selected: list[str]) -> dict:
    files = sorted(set(OWNERS + [item.split('::', 1)[0] for item in selected]))
    return {
        'base_commit': git(ROOT, 'rev-parse', 'HEAD'),
        'source_files': {item: digest(ROOT / item) for item in files if (ROOT / item).is_file()},
        'datasets_commit': git(DATASETS, 'rev-parse', 'HEAD'),
        'datasets_tracked_status': git(DATASETS, 'status', '--porcelain', '--untracked-files=no'),
        'retained_assets': {str(item): digest(item) for item in ASSETS if item.is_file()},
    }


def main() -> int:
    name, *selected = sys.argv[1:]
    if not selected or not name.replace('-', '').isalnum():
        raise ValueError('use a bounded run name followed by exact pytest selections')
    OUT.mkdir(parents=True, exist_ok=True)
    seal_path = Path('/dev/shm') / f'independent-review-{name}-{uuid.uuid4().hex}.duckdb'
    environment = dict(os.environ)
    overrides = {
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'IPFS_DATASETS_PY_MINIMAL_IMPORTS': '1',
        'PYTHONPATH': f'{ROOT}:{DATASETS}',
        'IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB': str(seal_path),
        'IPFS_ACCELERATE_PYTEST_SEAL': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'CUDA_VISIBLE_DEVICES': '',
    }
    environment.update(overrides)
    command = [sys.executable, '-u', '-B', '-m', 'pytest', '-c', 'pytest.ini',
               '--noconftest', '--import-mode=importlib', '-q', '--tb=short',
               '--show-capture=no', '--color=no', '-o', 'log_cli=false',
               f'--junitxml={OUT / (name + ".xml")}', *selected]
    before = snapshot(selected)
    (OUT / f'{name}-before.json').write_text(json.dumps(before, indent=2) + '\n')
    (OUT / f'{name}.command.sh').write_text(
        '#!/usr/bin/env bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) + '\n' +
        'env ' + ' '.join(shlex.quote(key + '=' + value) for key, value in overrides.items()) +
        ' ' + shlex.join(command) + '\n')
    print(f'{name}: start {len(selected)} exact selections; fresh seal {seal_path}', flush=True)
    started = time.monotonic()
    with (OUT / f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    after = snapshot(selected)
    (OUT / f'{name}-after.json').write_text(json.dumps(after, indent=2) + '\n')
    cases = {}
    xml_path = OUT / f'{name}.xml'
    if xml_path.is_file():
        for case in ET.parse(xml_path).iter('testcase'):
            status = 'passed'
            for kind in ('failure', 'error', 'skipped'):
                if case.find(kind) is not None:
                    status = kind
                    break
            cases[case.attrib.get('classname', '') + '::' + case.attrib.get('name', '')] = status
    seals = {}
    if seal_path.is_file():
        import duckdb
        connection = duckdb.connect(str(seal_path), read_only=True)
        try:
            count, minimum, maximum, authority = connection.execute(
                'SELECT count(*), min(file_count), max(file_count), bool_or(completion_authority) FROM pytest_ast_seal'
            ).fetchone()
            seals = {'records': count, 'closure_files_min': minimum,
                     'closure_files_max': maximum, 'any_completion_authority': bool(authority)}
        finally:
            connection.close()
    summary = {
        'command': command, 'environment_overrides': overrides,
        'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started,
        'pins_unchanged': before == after, 'ast_sealing': seals,
        'counts': {status: list(cases.values()).count(status) for status in ('passed', 'failure', 'error', 'skipped')},
        'case_outcomes': cases,
        'evidence_files': {path.name: digest(path) for path in (xml_path, OUT / f'{name}.log') if path.is_file()},
    }
    (OUT / f'{name}.qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ('returncode', 'elapsed_seconds', 'pins_unchanged', 'counts', 'ast_sealing')}), flush=True)
    return result.returncode or (0 if before == after else 90)


if __name__ == '__main__':
    raise SystemExit(main())
