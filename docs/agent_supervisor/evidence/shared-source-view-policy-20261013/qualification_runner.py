"""Run one frozen local qualification attempt; never overwrite prior evidence.

Invoke with the intended Python interpreter and ``qualification-01`` after the
source/test owners announce their freeze. This runner is not an admission to
providers, model inference/training, Docker, source execution or token savings.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
DATASETS = Path('/home/barberb/lift_coding/.worktrees/source-worker-datasets-20261012')
DATASETS_EXECUTION_REVISION = 'f47161b757190380e9898aff155067c1ad609217'
DATASETS_PARENT_PIN = 'b95938c2a816075466f8dc84f339cf622414030d'
ACCELERATOR_BASE = '85b21ded006fa9850ab49ee9f33bd320457a79a5'
TERMINAL_SITE = Path('/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages')
KIT = Path('/home/barberb/lift_coding/.worktrees/publication-main-20261004/kit')

TARGETS = (
    'test/api/test_shared_source_view_policy.py',
    'test/api/test_reversible_source_view.py',
    'test/api/test_repository_code_view_inventory.py',
    'test/api/test_repository_source_inventory.py',
    'test/api/test_worker_source_views.py',
    'test/api/test_worker_source_working_views.py',
    'test/api/test_source_view_edits.py',
    'test/api/test_source_view_edit_adversarial.py',
    'test/api/test_source_view_edit_candidate.py',
    'test/api/test_source_view_edit_formal_refresh.py',
    'test/api/test_worker_candidate_context.py',
    'test/api/test_worker_candidate_context_admission.py',
    'test/api/test_worker_candidate_context_client.py',
    'test/api/test_worker_candidate_context_broker.py',
    'test/api/test_worker_source_client.py',
    'test/api/test_worker_source_working_client.py',
    'test/api/test_worker_source_broker.py',
    'test/api/test_worker_source_working_broker.py',
    'benchmarks/agent_supervisor/container_coding/test_source_view_worker_wiring.py',
    'benchmarks/agent_supervisor/container_coding/test_source_view_trace_comparison.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_candidate_context_observation.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_worker_capability.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_deployment.py',
)
ACCELERATOR_SOURCES = (
    'ipfs_accelerate_py/__init__.py',
    *('ipfs_accelerate_py/agent_supervisor/runtime/' + name + '.py' for name in (
        'source_view_policy', 'repository_code_view_inventory', 'repository_source_inventory',
        'worker_source_views', 'source_view_edit_candidate', 'reversible_source_view',
        'source_view_edits', 'worker_source_client', 'worker_source_broker',
        'worker_candidate_context', 'worker_candidate_context_client', 'worker_candidate_artifacts',
        'worker_candidate_selection', 'source_formalization_context', 'numerical_obligation_broker',
        'numerical_obligation_client', 'doctor_candidate_runner', 'semantic_context_runtime',
        'semantic_capsule_selection', 'semantic_router_translation', 'router_implementation_runner',
        'supervisor_meta_index',
    )),
    *('ipfs_accelerate_py/agent_supervisor/semantic_state/' + name + '.py' for name in (
        'worktree', 'contracts', 'scheduling_contracts', 'datasets_adapter', 'wire',
    )),
    'ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py',
    'ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py',
    'ipfs_accelerate_py/testing/pytest_ast_seal.py',
    *('benchmarks/agent_supervisor/container_coding/' + name + '.py' for name in (
        'source_view_trace_comparison', 'container_worker_deployment',
        'terminal_source_view_observation', 'test_terminal_source_view_observation',
        'terminal_output_observation', 'terminal_numerical_retrieval_observation',
        'terminal_worker_capability', 'benchmark_resource_profile', 'terminal_deployment',
        'full_supervisor_benchmark', 'full_supervisor_harbor_agent', 'test_terminal_deployment',
    )),
    'test/api/test_semantic_router_integration.py',
    'test/api/test_semantic_router_translation.py',
    'test/api/test_numerical_obligation_client.py',
    'test/common/fixtures.py',
    'conftest.py', 'test/conftest.py', 'test/api/conftest.py', 'pytest.ini',
)
DATASETS_SOURCES = (
    'ipfs_datasets_py/__init__.py',
    *('ipfs_datasets_py/logic/software_contracts/' + name + '.py' for name in (
        'cache', 'content', 'python_frontend', 'semantic_index/committed_snapshot',
        'semantic_index/chunked_snapshot', 'semantic_index/paged_snapshot',
        'semantic_index/scanner', 'semantic_index/snapshot', 'semantic_index/python_analysis',
        'semantic_index/pytest_analysis', 'semantic_index/symbol_graph',
        'semantic_index/identity', 'semantic_index/models', 'semantic_state/api',
        'semantic_state/models', 'semantic_state/capsules', 'semantic_state/bindings',
        'semantic_state/merkle',
    )),
    *('ipfs_datasets_py/logic/software_verification/' + name + '.py' for name in (
        'numerical_logic', 'numerical_source', 'numerical_obligation_retrieval',
        'source_adapters', 'syntax_bridge', 'program',
    )),
    'ipfs_datasets_py/logic/backends/smt/compiler.py',
    'ipfs_datasets_py/logic/families/registry.py',
    'ipfs_datasets_py/logic/syntax_core/ast.py',
    'ipfs_datasets_py/logic/syntax_core/signatures.py',
)
UNCHANGED_CODEC_SOURCES = (
    'ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/source_view_edits.py',
)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repository, *args):
    return subprocess.check_output(['git', *args], cwd=repository)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def manifest(paths):
    rows = []
    for repository, checkout, names in (
        ('accelerator', ROOT, paths), ('datasets_execution_checkout', DATASETS, DATASETS_SOURCES),
    ):
        for name in sorted(set(names)):
            raw = (checkout / name).read_bytes()
            row = dict(repository=repository, path=name, bytes=len(raw), sha256=digest(raw))
            if checkout == DATASETS:
                expected = git(DATASETS, 'show', DATASETS_EXECUTION_REVISION + ':' + name)
                row['git_object_sha256'] = digest(expected)
                row['matches_immutable_git_object'] = raw == expected
            rows.append(row)
    return rows


def main():
    if len(sys.argv) != 2 or re.fullmatch(r'qualification-[0-9]{2,3}', sys.argv[1]) is None:
        raise SystemExit('Usage: <python> qualification_runner.py qualification-01')
    if git(DATASETS, 'rev-parse', 'HEAD').decode().strip() != DATASETS_EXECUTION_REVISION:
        raise SystemExit('Refusing another datasets execution checkout revision')
    for directory in (TERMINAL_SITE, KIT):
        if not directory.is_dir():
            raise SystemExit('Required existing dependency path missing: ' + str(directory))
    names = set((*ACCELERATOR_SOURCES, *TARGETS, str(Path(__file__).resolve().relative_to(ROOT))))
    # Bind every changed local Python file as well as the explicit producer/test
    # list. This does not expand the test population or import those modules.
    for args in (('diff', '--name-only', 'HEAD', '--', '*.py'),
                 ('ls-files', '--others', '--exclude-standard', '--', '*.py')):
        names.update(git(ROOT, *args).decode().splitlines())
    before = manifest(names)  # Missing not-yet-frozen sources refuse before output creation.
    if not all(row.get('matches_immutable_git_object', True) for row in before):
        raise SystemExit('Datasets execution source differs from its immutable Git object')
    legacy = []
    for name in UNCHANGED_CODEC_SOURCES:
        expected = git(ROOT, 'show', ACCELERATOR_BASE + ':' + name)
        row = dict(path=name, baseline_revision=ACCELERATOR_BASE,
                   baseline_sha256=digest(expected), unchanged=(ROOT / name).read_bytes() == expected)
        legacy.append(row)
    if not all(row['unchanged'] for row in legacy):
        raise SystemExit('Historical codec/pure mapper bytes changed; qualification requires review')
    attempt = PHASE / sys.argv[1]
    attempt.mkdir(exist_ok=False)
    runtime = Path(tempfile.mkdtemp(prefix='shared-source-view-policy-' + sys.argv[1] + '-'))
    receipt = attempt / 'commands.json'
    updates = dict(
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='0',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0',
        IPFS_KIT_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(map(str, (ROOT, DATASETS, TERMINAL_SITE, KIT))),
    )
    removed = ('IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET', 'IPFS_ACCELERATE_AGENT_QUACK_TOKEN',
               'IPFS_ACCELERATE_META_INDEX_DUCKDB', 'IPFS_ACCELERATE_META_INDEX_DUCKLAKE',
               'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'IPFS_TEST_PROOF_REUSE_MODE')
    record = dict(
        schema='shared-source-view-policy-qualification-command@1', started_at_utc=utc(),
        cwd=str(ROOT), attempt_directory=str(attempt), interpreter=sys.executable,
        resolved_interpreter=str(Path(sys.executable).resolve()), python_version=platform.python_version(),
        private_runtime_directory=str(runtime), runtime_databases_in_public_evidence=False,
        targets=list(TARGETS), accelerator_base_revision=ACCELERATOR_BASE,
        accelerator_execution_head=git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
        kit_path=str(KIT), kit_execution_head=git(KIT, 'rev-parse', 'HEAD').decode().strip(),
        datasets_path=str(DATASETS), datasets_execution_revision=DATASETS_EXECUTION_REVISION,
        datasets_parent_pin_reference_only=DATASETS_PARENT_PIN,
        datasets_parent_pin_executed=False, before=before,
        source_binding_scope='listed_local_files_plus_changed_python_not_full_toolchain_attestation',
        unchanged_historical_codec_and_pure_mapper=legacy, environment_updates=updates,
        removed_inherited_environment_names=list(removed), invocations=[],
        provider_calls=0, model_calls=0, training_steps=0, solver_calls=0, docker_run=False,
        provider_tokens=None, total_token_savings_measured=False, official_task_result=None,
        proof_authority=False, completion_authority=False,
    )
    write_json(receipt, record)
    failure = None
    try:
        for phase in ('collection', 'tests'):
            state = runtime / (phase + '-state')
            state.mkdir()
            env = os.environ.copy()
            for key in removed:
                env.pop(key, None)
            phase_updates = dict(updates,
                IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(state / 'orchestration'),
                IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(state / 'seal.duckdb'))
            env.update(phase_updates)
            argv = [sys.executable, '-B', '-m', 'pytest', '-c', 'pytest.ini', *TARGETS,
                    '--color=no', '-o', 'log_cli=false', '-o', 'cache_dir=' + str(state / 'pytest-cache')]
            if phase == 'collection':
                argv += ['--collect-only', '-qq']
            else:
                argv += ['--maxfail=1', '-q', '--tb=short', '--junitxml=' + str(attempt / 'junit.xml')]
            stdout = attempt / (phase + '.stdout.log')
            stderr = attempt / (phase + '.stderr.log')
            call = dict(phase=phase, argv=argv, cwd=str(ROOT), environment_updates=phase_updates,
                        started_at_utc=utc(), stdout_path=str(stdout), stderr_path=str(stderr))
            record['invocations'].append(call)
            write_json(receipt, record)
            print('Starting ' + phase + ': ' + str(attempt), flush=True)
            started = time.monotonic()
            with stdout.open('wb') as out, stderr.open('wb') as err:
                result = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err)
            call.update(exit_code=result.returncode, completed_at_utc=utc(),
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        stdout_sha256=digest(stdout.read_bytes()), stdout_bytes=stdout.stat().st_size,
                        stderr_sha256=digest(stderr.read_bytes()), stderr_bytes=stderr.stat().st_size)
            if phase == 'collection':
                ids = [line.strip() for line in stdout.read_text(errors='replace').splitlines()
                       if line.startswith(('test/api/', 'benchmarks/')) and '.py::' in line]
                record['collected_tests'] = len(ids)
                record['collected_distinct_tests'] = len(set(ids))
                write_json(attempt / 'nodeids.json', dict(schema='pytest-qualified-nodeids@1',
                           nodeids=ids, count=len(ids), distinct_count=len(set(ids))))
                if result.returncode == 0 and (not ids or len(ids) != len(set(ids))):
                    raise RuntimeError('Collection did not produce a nonempty unique node population')
            write_json(receipt, record)
            if result.returncode:
                break
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        record['runner_error'] = failure
    finally:
        record['completed_at_utc'] = utc()
        try:
            record['after'] = manifest(names)
            record['source_before_after_equal'] = before == record['after']
            record['datasets_execution_head_after'] = git(DATASETS, 'rev-parse', 'HEAD').decode().strip()
            record['kit_execution_head_after'] = git(KIT, 'rev-parse', 'HEAD').decode().strip()
            record['kit_head_before_after_equal'] = record['kit_execution_head'] == record['kit_execution_head_after']
        except Exception as error:
            record['final_source_observation_error'] = dict(type=type(error).__name__, message=str(error))
            record['source_before_after_equal'] = False
        junit = attempt / 'junit.xml'
        if junit.exists():
            record['junit_sha256'] = digest(junit.read_bytes())
            try:
                document = ET.parse(junit).getroot()
                suites = [document] if document.tag == 'testsuite' else document.findall('testsuite')
                record['junit'] = {key: sum(int(suite.get(key, 0)) for suite in suites)
                                   for key in ('tests', 'failures', 'errors', 'skipped')}
                record['junit_testcase_count'] = sum(1 for _ in document.iter('testcase'))
            except Exception as error:
                record['junit_observation_error'] = dict(type=type(error).__name__, message=str(error))
        record['collection_execution_count_agrees'] = (
            record.get('collected_distinct_tests') == record.get('junit', {}).get('tests')
            == record.get('junit_testcase_count') and bool(record.get('collected_distinct_tests')))
        record['status'] = 'pass' if (
            failure is None and len(record['invocations']) == 2
            and all(call.get('exit_code') == 0 for call in record['invocations'])
            and record['source_before_after_equal'] and record['collection_execution_count_agrees']
            and record.get('datasets_execution_head_after') == DATASETS_EXECUTION_REVISION
            and record.get('kit_head_before_after_equal') is True
            and all(record['junit'].get(key) == 0 for key in ('failures', 'errors', 'skipped'))
        ) else 'fail'
        write_json(receipt, record)
    print(json.dumps({key: record.get(key) for key in (
        'status', 'collected_distinct_tests', 'junit', 'source_before_after_equal',
        'collection_execution_count_agrees', 'runner_error')}, sort_keys=True), flush=True)
    return 0 if record['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
