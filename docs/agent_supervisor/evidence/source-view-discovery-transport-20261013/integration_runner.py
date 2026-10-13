"""One full merged qualification pytest invocation with frozen, bounded evidence.

Install into the phase evidence directory only after the final source freeze.
No separate collection invocation, provider/model/training/Docker run, token
benefit or complete dependency-closure attestation is authorized by this runner.
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
DATASETS = Path('/home/barberb/lift_coding/.worktrees/source-discovery-datasets-20261013')
DATASETS_EXECUTION_REVISION = 'afadfbcf8d33a67568c133bb761c43d42a9a133c'
DATASETS_PARENT_PIN = 'afadfbcf8d33a67568c133bb761c43d42a9a133c'
PARENT_INTEGRATION_REVISION_REFERENCE_ONLY = 'b69fe600ee350d4c016cbc3dd889265963e6e903'
ACCELERATOR_BASE = '8aafb960fd634d218478376aa18bc83f8b121dce'
FULL_RUNNER_SHA256 = '187a352e3252fe53286f900511476f86cb41af60993b25af7489f2e918798b24'
TERMINAL_SITE = Path('/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages')
KIT = Path('/home/barberb/lift_coding/.worktrees/publication-main-20261004/kit')

TARGETS = ('test/api/test_worker_source_discovery_transport.py', 'test/api/test_shared_source_view_policy.py', 'test/api/test_reversible_source_view.py', 'test/api/test_repository_code_view_inventory.py', 'test/api/test_repository_source_inventory.py', 'test/api/test_worker_source_views.py', 'test/api/test_worker_source_working_views.py', 'test/api/test_source_view_edits.py', 'test/api/test_source_view_edit_adversarial.py', 'test/api/test_source_view_edit_candidate.py', 'test/api/test_source_view_edit_formal_refresh.py', 'test/api/test_worker_candidate_context.py', 'test/api/test_worker_candidate_context_admission.py', 'test/api/test_worker_candidate_context_client.py', 'test/api/test_worker_candidate_context_broker.py', 'test/api/test_worker_source_client.py', 'test/api/test_worker_source_working_client.py', 'test/api/test_worker_source_broker.py', 'test/api/test_worker_source_working_broker.py', 'benchmarks/agent_supervisor/container_coding/test_source_view_worker_wiring.py', 'benchmarks/agent_supervisor/container_coding/test_source_view_trace_comparison.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_candidate_context_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_worker_capability.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_deployment.py', 'test/api/test_numerical_obligation_client.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_broker.py', 'benchmarks/agent_supervisor/container_coding/test_numerical_obligation_worker_wiring.py', 'test/api/test_numerical_obligation_router.py', 'test/api/test_semantic_router_translation.py', 'test/api/test_semantic_router_integration.py', 'test/api/test_source_lookup_worker_router.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_observation.py', 'test/api/test_context_selection_observation.py', 'test/api/test_context_selection_preparation.py', 'benchmarks/agent_supervisor/container_coding/test_context_projection_trace_comparison.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_output_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_policy.py', 'benchmarks/agent_supervisor/container_coding/test_benchmark_comparison.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_intent_action_readiness.py')

ACCELERATOR_SOURCES = ('benchmarks/agent_supervisor/container_coding/benchmark_comparison.py', 'benchmarks/agent_supervisor/container_coding/benchmark_controls.py', 'benchmarks/agent_supervisor/container_coding/benchmark_provider_profile.py', 'benchmarks/agent_supervisor/container_coding/benchmark_resource_profile.py', 'benchmarks/agent_supervisor/container_coding/container_worker_deployment.py', 'benchmarks/agent_supervisor/container_coding/context_projection_trace_comparison.py', 'benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py', 'benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py', 'benchmarks/agent_supervisor/container_coding/source_view_trace_comparison.py', 'benchmarks/agent_supervisor/container_coding/terminal_context_audit.py', 'benchmarks/agent_supervisor/container_coding/terminal_deployment.py', 'benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py', 'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py', 'benchmarks/agent_supervisor/container_coding/terminal_intent_action_readiness.py', 'benchmarks/agent_supervisor/container_coding/terminal_native_progress.py', 'benchmarks/agent_supervisor/container_coding/terminal_numerical_retrieval_observation.py', 'benchmarks/agent_supervisor/container_coding/terminal_output_observation.py', 'benchmarks/agent_supervisor/container_coding/terminal_source_view_observation.py', 'benchmarks/agent_supervisor/container_coding/terminal_source_view_policy.py', 'benchmarks/agent_supervisor/container_coding/terminal_worker_capability.py', 'benchmarks/agent_supervisor/container_coding/test_benchmark_comparison.py', 'benchmarks/agent_supervisor/container_coding/test_context_projection_trace_comparison.py', 'benchmarks/agent_supervisor/container_coding/test_numerical_obligation_worker_wiring.py', 'benchmarks/agent_supervisor/container_coding/test_source_view_trace_comparison.py', 'benchmarks/agent_supervisor/container_coding/test_source_view_worker_wiring.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_candidate_context_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_deployment.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_intent_action_384_transport.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_intent_action_readiness.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_context.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_broker.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_retrieval_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_output_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_semantic_metadata_policy.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_policy.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_worker_capability.py', 'conftest.py', 'docs/agent_supervisor/evidence/source-view-discovery-transport-20261013/qualification_runner.py', 'ipfs_accelerate_py/__init__.py', 'ipfs_accelerate_py/agent_supervisor/context/context_compiler.py', 'ipfs_accelerate_py/agent_supervisor/context/context_contracts.py', 'ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py', 'ipfs_accelerate_py/agent_supervisor/runtime/coding_reply_contract.py', 'ipfs_accelerate_py/agent_supervisor/runtime/container_worker_boundary.py', 'ipfs_accelerate_py/agent_supervisor/runtime/context_selection_observation.py', 'ipfs_accelerate_py/agent_supervisor/runtime/doctor_candidate_runner.py', 'ipfs_accelerate_py/agent_supervisor/runtime/intent_384_advisor.py', 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py', 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_broker.py', 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py', 'ipfs_accelerate_py/agent_supervisor/runtime/repository_code_view_inventory.py', 'ipfs_accelerate_py/agent_supervisor/runtime/repository_source_inventory.py', 'ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py', 'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py', 'ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_selection.py', 'ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py', 'ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py', 'ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py', 'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py', 'ipfs_accelerate_py/agent_supervisor/runtime/source_view_edit_candidate.py', 'ipfs_accelerate_py/agent_supervisor/runtime/source_view_edits.py', 'ipfs_accelerate_py/agent_supervisor/runtime/source_view_policy.py', 'ipfs_accelerate_py/agent_supervisor/runtime/supervisor_meta_index.py', 'ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_artifacts.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_context.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_context_client.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_selection.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_source_broker.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py', 'ipfs_accelerate_py/agent_supervisor/semantic_state/contracts.py', 'ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py', 'ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_database.py', 'ipfs_accelerate_py/agent_supervisor/semantic_state/scheduling_contracts.py', 'ipfs_accelerate_py/agent_supervisor/semantic_state/wire.py', 'ipfs_accelerate_py/agent_supervisor/semantic_state/worktree.py', 'ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py', 'ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py', 'ipfs_accelerate_py/mcp_server/mcplusplus/kubo_cid.py', 'ipfs_accelerate_py/testing/pytest_ast_seal.py', 'pytest.ini', 'test/api/conftest.py', 'test/api/test_context_selection_observation.py', 'test/api/test_context_selection_preparation.py', 'test/api/test_numerical_obligation_client.py', 'test/api/test_numerical_obligation_router.py', 'test/api/test_repository_code_view_inventory.py', 'test/api/test_repository_source_inventory.py', 'test/api/test_reversible_source_view.py', 'test/api/test_semantic_router_integration.py', 'test/api/test_semantic_router_translation.py', 'test/api/test_shared_source_view_policy.py', 'test/api/test_source_formalization_context.py', 'test/api/test_source_formalization_router.py', 'test/api/test_source_lookup_worker_router.py', 'test/api/test_source_view_edit_adversarial.py', 'test/api/test_source_view_edit_candidate.py', 'test/api/test_source_view_edit_formal_refresh.py', 'test/api/test_source_view_edits.py', 'test/api/test_worker_candidate_context.py', 'test/api/test_worker_candidate_context_admission.py', 'test/api/test_worker_candidate_context_broker.py', 'test/api/test_worker_candidate_context_client.py', 'test/api/test_worker_source_broker.py', 'test/api/test_worker_source_client.py', 'test/api/test_worker_source_discovery_transport.py', 'test/api/test_worker_source_views.py', 'test/api/test_worker_source_working_broker.py', 'test/api/test_worker_source_working_client.py', 'test/api/test_worker_source_working_views.py', 'test/common/fixtures.py', 'test/conftest.py')

DATASETS_SOURCES = ('ipfs_datasets_py/__init__.py', 'ipfs_datasets_py/logic/autoformal/tree_pin.py', 'ipfs_datasets_py/logic/backends/smt/compiler.py', 'ipfs_datasets_py/logic/families/registry.py', 'ipfs_datasets_py/logic/formalization/autoencoder/intent_action_checkpoint_384.py', 'ipfs_datasets_py/logic/formalization/autoencoder/source_embeddings_384.py', 'ipfs_datasets_py/logic/formalization/autoencoder/source_training_v2.py', 'ipfs_datasets_py/logic/formalization/autoencoder/structured_source_384.py', 'ipfs_datasets_py/logic/intent_ir/decoder.py', 'ipfs_datasets_py/logic/intent_ir/formalize/action_contracts.py', 'ipfs_datasets_py/logic/intent_ir/formalize/rich_grammar.py', 'ipfs_datasets_py/logic/intent_ir/schema.py', 'ipfs_datasets_py/logic/ir_core/canonical.py', 'ipfs_datasets_py/logic/ir_core/schema_registry.py', 'ipfs_datasets_py/logic/legal_ir/canonical_contracts.py', 'ipfs_datasets_py/logic/security_ir/model.py', 'ipfs_datasets_py/logic/software_contracts/cache.py', 'ipfs_datasets_py/logic/software_contracts/content.py', 'ipfs_datasets_py/logic/software_contracts/python_frontend.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/chunked_snapshot.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/committed_snapshot.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/identity.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/models.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/paged_snapshot.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/pytest_analysis.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/python_analysis.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/scanner.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/snapshot.py', 'ipfs_datasets_py/logic/software_contracts/semantic_index/symbol_graph.py', 'ipfs_datasets_py/logic/software_contracts/semantic_state/api.py', 'ipfs_datasets_py/logic/software_contracts/semantic_state/bindings.py', 'ipfs_datasets_py/logic/software_contracts/semantic_state/capsules.py', 'ipfs_datasets_py/logic/software_contracts/semantic_state/merkle.py', 'ipfs_datasets_py/logic/software_contracts/semantic_state/models.py', 'ipfs_datasets_py/logic/software_verification/numerical_logic.py', 'ipfs_datasets_py/logic/software_verification/numerical_obligation_retrieval.py', 'ipfs_datasets_py/logic/software_verification/numerical_source.py', 'ipfs_datasets_py/logic/software_verification/program.py', 'ipfs_datasets_py/logic/software_verification/source_adapters.py', 'ipfs_datasets_py/logic/software_verification/syntax_bridge.py', 'ipfs_datasets_py/logic/syntax_core/ast.py', 'ipfs_datasets_py/logic/syntax_core/signatures.py', 'ipfs_datasets_py/logic/ui_ux_ir/decoder.py', 'ipfs_datasets_py/logic/ui_ux_ir/schema.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_embedding_production.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_embedding_runtime.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/domain_384_autoencoder.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_formula_codec.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_latent_formula.py')

UNCHANGED_CODEC_SOURCES = ('ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py', 'ipfs_accelerate_py/agent_supervisor/runtime/source_view_edits.py')

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
    if len(sys.argv) != 2 or re.fullmatch(r'integration-[0-9]{2,3}', sys.argv[1]) is None:
        raise SystemExit('Usage: <python> integration_runner.py integration-01')
    if git(DATASETS, 'rev-parse', 'HEAD').decode().strip() != DATASETS_EXECUTION_REVISION:
        raise SystemExit('Refusing another datasets execution checkout revision')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', ACCELERATOR_BASE, 'HEAD'], cwd=ROOT).returncode:
        raise SystemExit('Incoming integration base is not retained in execution HEAD')
    full_runner = PHASE / 'qualification_runner.py'
    if digest(full_runner.read_bytes()) != FULL_RUNNER_SHA256:
        raise SystemExit('Historical full qualification runner changed')
    for directory in (TERMINAL_SITE, KIT):
        if not directory.is_dir():
            raise SystemExit('Required existing dependency path missing: ' + str(directory))
    names = set((*ACCELERATOR_SOURCES, *TARGETS, str(Path(__file__).resolve().relative_to(ROOT))))
    before = manifest(names)
    if not all(row.get('matches_immutable_git_object', True) for row in before):
        raise SystemExit('Datasets execution source differs from its immutable Git object')
    legacy = []
    for name in UNCHANGED_CODEC_SOURCES:
        expected = git(ROOT, 'show', ACCELERATOR_BASE + ':' + name)
        legacy.append(dict(path=name, baseline_revision=ACCELERATOR_BASE,
            baseline_sha256=digest(expected), unchanged=(ROOT / name).read_bytes() == expected))
    if not all(row['unchanged'] for row in legacy):
        raise SystemExit('Historical codec/pure mapper bytes changed; integration requires review')
    attempt = PHASE / sys.argv[1]
    attempt.mkdir(exist_ok=False)
    runtime = Path(tempfile.mkdtemp(prefix='source-view-discovery-transport-' + sys.argv[1] + '-'))
    state = runtime / 'tests-state'
    state.mkdir()
    updates = dict(
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='0',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0',
        IPFS_KIT_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(map(str, (ROOT, DATASETS, TERMINAL_SITE, KIT))),
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(state / 'orchestration'),
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(state / 'seal.duckdb'))
    removed = ('IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET', 'IPFS_ACCELERATE_AGENT_QUACK_TOKEN',
               'IPFS_ACCELERATE_META_INDEX_DUCKDB', 'IPFS_ACCELERATE_META_INDEX_DUCKLAKE',
               'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'IPFS_TEST_PROOF_REUSE_MODE')
    env = os.environ.copy()
    for key in removed:
        env.pop(key, None)
    env.update(updates)
    argv = [sys.executable, '-B', '-m', 'pytest', '-c', 'pytest.ini', *TARGETS,
            '--color=no', '-o', 'log_cli=false', '-o', 'cache_dir=' + str(state / 'pytest-cache'),
            '--tb=short', '--junitxml=' + str(attempt / 'junit.xml')]
    stdout, stderr = attempt / 'tests.stdout.log', attempt / 'tests.stderr.log'
    call = dict(phase='integration_tests', argv=argv, cwd=str(ROOT), environment_updates=updates,
        started_at_utc=utc(), stdout_path=str(stdout), stderr_path=str(stderr))
    record = dict(
        schema='source-view-discovery-transport-integration-command@1', started_at_utc=utc(),
        cwd=str(ROOT), attempt_directory=str(attempt), interpreter=sys.executable,
        resolved_interpreter=str(Path(sys.executable).resolve()), python_version=platform.python_version(),
        private_runtime_directory=str(runtime), runtime_databases_in_public_evidence=False,
        targets=list(TARGETS), pytest_invocations=1, separate_collection_invocation=False,
        accelerator_base_revision=ACCELERATOR_BASE,
        accelerator_execution_head=git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
        kit_path=str(KIT), kit_execution_head=git(KIT, 'rev-parse', 'HEAD').decode().strip(),
        datasets_path=str(DATASETS), datasets_execution_revision=DATASETS_EXECUTION_REVISION,
        datasets_parent_pin_reference=DATASETS_PARENT_PIN,
        datasets_parent_pin_executed=True,
        previous_qualification_datasets_revision='f47161b757190380e9898aff155067c1ad609217',
        previous_f471_qualification_evidence_modified=False,
        parent_integration_revision_reference_only=PARENT_INTEGRATION_REVISION_REFERENCE_ONLY,
        source_list_seed='qualification-02/commands.json before list; no failed prior result is reused as a merged pass',
        historical_full_runner_sha256=digest(full_runner.read_bytes()), before=before,
        source_binding_scope='listed_dependency_rows_not_complete_import_closure_or_full_toolchain_attestation',
        unchanged_historical_codec_and_pure_mapper=legacy, environment_updates=updates,
        removed_inherited_environment_names=list(removed), invocations=[call],
        provider_calls=0, model_calls=0, training_steps=0, solver_calls=0, docker_run=False,
        provider_tokens=None, total_token_savings_measured=False, official_task_result=None,
        proof_authority=False, completion_authority=False)
    receipt = attempt / 'commands.json'
    write_json(receipt, record)
    failure = None
    print('Starting one integration pytest invocation: ' + str(attempt), flush=True)
    started = time.monotonic()
    try:
        with stdout.open('wb') as out, stderr.open('wb') as err:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err)
        call.update(exit_code=result.returncode, completed_at_utc=utc(),
            elapsed_seconds=round(time.monotonic() - started, 3))
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        record['runner_error'] = failure
    finally:
        record['completed_at_utc'] = utc()
        for prefix, path in (('stdout', stdout), ('stderr', stderr)):
            if path.exists():
                raw = path.read_bytes()
                call.update({prefix + '_sha256': digest(raw), prefix + '_bytes': len(raw)})
        try:
            record['after'] = manifest(names)
            record['source_before_after_equal'] = before == record['after']
            record['datasets_execution_head_after'] = git(DATASETS, 'rev-parse', 'HEAD').decode().strip()
            record['kit_execution_head_after'] = git(KIT, 'rev-parse', 'HEAD').decode().strip()
            record['kit_head_before_after_equal'] = record['kit_execution_head'] == record['kit_execution_head_after']
            record['historical_full_runner_unchanged'] = digest(full_runner.read_bytes()) == FULL_RUNNER_SHA256
        except Exception as error:
            record['final_source_observation_error'] = dict(type=type(error).__name__, message=str(error))
            record['source_before_after_equal'] = False
        if stdout.exists():
            collected = re.findall(r'^collected ([0-9]+) items?\s*$', stdout.read_text(errors='replace'), re.M)
            record['collected_count_observations'] = list(map(int, collected))
            record['collected_tests'] = int(collected[0]) if len(collected) == 1 else None
        junit = attempt / 'junit.xml'
        if junit.exists():
            record['junit_sha256'] = digest(junit.read_bytes())
            try:
                document = ET.parse(junit).getroot()
                suites = [document] if document.tag == 'testsuite' else document.findall('testsuite')
                record['junit'] = {key: sum(int(suite.get(key, 0)) for suite in suites)
                    for key in ('tests', 'failures', 'errors', 'skipped')}
                cases = [(item.get('classname'), item.get('name')) for item in document.iter('testcase')]
                record['junit_testcase_count'] = len(cases)
                record['junit_distinct_testcase_count'] = len(set(cases))
                record['junit_case_identity'] = '(classname, name)'
            except Exception as error:
                record['junit_observation_error'] = dict(type=type(error).__name__, message=str(error))
        record['collection_execution_count_agrees'] = (
            record.get('collected_tests') == record.get('junit', {}).get('tests')
            == record.get('junit_testcase_count') == record.get('junit_distinct_testcase_count')
            and bool(record.get('collected_tests')))
        record['status'] = 'pass' if (
            failure is None and call.get('exit_code') == 0
            and record.get('source_before_after_equal') is True
            and record.get('collection_execution_count_agrees') is True
            and record.get('datasets_execution_head_after') == DATASETS_EXECUTION_REVISION
            and record.get('kit_head_before_after_equal') is True
            and record.get('historical_full_runner_unchanged') is True
            and all(record.get('junit', {}).get(key) == 0 for key in ('failures', 'errors', 'skipped'))
        ) else 'fail'
        write_json(receipt, record)
    print(json.dumps({key: record.get(key) for key in (
        'status', 'collected_tests', 'junit', 'junit_distinct_testcase_count',
        'source_before_after_equal', 'collection_execution_count_agrees', 'runner_error')}, sort_keys=True), flush=True)
    return 0 if record['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
