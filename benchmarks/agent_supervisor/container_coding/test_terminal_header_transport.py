"""Exact header profile transport with authored model bytes and reviewed intent."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_transport import selected, build
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as owner


@pytest.fixture
def header(selected, tmp_path):
    instruction = tmp_path / 'instruction.md'; instruction.write_text('Repair unsafe headers.')
    contract = _requirements(instruction, symbolic=True)
    selector = dict(schema=owner.SCHEMA, review_ref='review:authored-transport@1',
        operation_id=contract['symbolic_operations']['operations'][0]['operation_id'],
        operator_id=owner.OPERATOR, source_path='bottle.py',
        protocol={'review_ref': 'review:authored-protocol@1', 'callback_parameter': 'start_response'},
        checker_profile=owner.CHECKER_PROFILE, **owner.FALSE)
    contract.update(schema=owner.CONTRACT_SCHEMA, source_applicability=selector)
    profile = dict(schema=owner.PROFILE_SCHEMA, selector_cid=cid_for_dag_json(selector),
        checker_profile=owner.CHECKER_PROFILE,
        solver_sha256=hashlib.sha256(Path(shutil.which('z3')).resolve().read_bytes()).hexdigest())
    selected[0].update(schema='terminal-source384-config@2', header_applicability=profile)
    selected[1].write_text(json.dumps(selected[0]))
    return contract, profile


def test_archive_preserves_reviewed_profile_and_checks_installed_solver(tmp_path, selected, header, capsys):
    contract, profile = header
    manifest = build(tmp_path, selected)
    binding = deployment.validate_source384_binding(manifest)
    assert binding['config']['header_applicability'] == profile
    deployment.verify_source384_archive(tmp_path / 'bundle/runtime.tar.gz', manifest)
    benchmark.validate_header_planning_selection(binding, contract, 'full')
    assert benchmark._intent_selection(contract)['planning_strategy'] == 'intent_symbolic'
    probe = deployment._source384_header_checker_probe(binding)
    exec(compile(probe, 'runtime-checker-preflight', 'exec'), {})
    observed = json.loads(capsys.readouterr().out)
    assert observed['sha256'] == profile['solver_sha256'] and observed['bytes'] > 0
    bad = deepcopy(binding); bad['config']['header_applicability']['solver_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='solver identity'):
        exec(compile(deployment._source384_header_checker_probe(bad), 'bad-checker-pin', 'exec'), {})


@pytest.mark.parametrize('change', ['no-index', 'missing-contract', 'missing-profile', 'wrong-selector', 'wrong-protocol'])
def test_selection_mismatch_refuses_before_container_calls(tmp_path, selected, header, monkeypatch, change):
    contract, _ = header
    if change == 'missing-profile':
        selected[0].pop('header_applicability'); selected[0]['schema'] = 'terminal-source384-config@1'
    if change == 'missing-contract': contract = None
    elif change == 'wrong-selector': contract['source_applicability']['review_ref'] = 'review:different@1'
    elif change == 'wrong-protocol': contract['source_applicability']['protocol']['callback_parameter'] = 'emit'
    build(tmp_path, selected)
    deploy = AsyncMock(); monkeypatch.setattr(adapter, 'deploy_supervisor', deploy)
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / 'logs', model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path / 'bundle'), arm='no-index' if change == 'no-index' else 'full',
        intent_requirement_contract=contract)
    with pytest.raises(ValueError, match='header'):
        asyncio.run(agent.setup(SimpleNamespace()))
    deploy.assert_not_awaited()


def test_legacy_selection_has_no_header_probe(tmp_path, selected):
    binding = deployment.validate_source384_binding(build(tmp_path, selected))
    assert deployment._source384_header_checker_probe(binding) is None
    benchmark.validate_header_planning_selection(binding, None, 'no-index')


@pytest.mark.parametrize('mutation', ['unknown-field', 'missing-profile', 'profile-in-v1', 'wrong-family'])
def test_closed_profile_schema_cannot_silently_fall_back(selected, header, mutation):
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_config import validate_source384_config
    config = deepcopy(selected[0])
    if mutation == 'unknown-field': config['header_applicability']['captured_head'] = {}
    elif mutation == 'missing-profile': config.pop('header_applicability')
    elif mutation == 'profile-in-v1': config['schema'] = 'terminal-source384-config@1'
    else: config['header_applicability']['checker_profile'] = 'unchecked@1'
    with pytest.raises(ValueError): validate_source384_config(config)


def test_direct_run_rejects_mismatch_before_upload(tmp_path, selected, header):
    contract, _ = header
    build(tmp_path, selected)
    contract['source_applicability']['review_ref'] = 'review:changed@1'
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / 'logs', model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path / 'bundle'), arm='full', intent_requirement_contract=contract)
    environment = SimpleNamespace(upload_file=AsyncMock())
    with pytest.raises(ValueError, match='header'):
        asyncio.run(agent.run('Repair unsafe headers.', environment, SimpleNamespace()))
    environment.upload_file.assert_not_awaited()


def test_qualifier_forwards_reviewed_intent_into_native_prepare(tmp_path, selected, header):
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualifier
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_PROFILE
    contract, _ = header
    manifest = build(tmp_path, selected)
    task = tmp_path / 'task'; task.mkdir()
    (task / 'instruction.md').write_text('Repair unsafe headers.')
    output = tmp_path / 'output'; output.mkdir()
    # Stop at download: exercise production transport without simulating a qualified result.
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(return_value=
        SimpleNamespace(stdout='', stderr='', return_code=1)),
        download_file=AsyncMock(side_effect=RuntimeError('transport stop')))
    with pytest.raises(RuntimeError, match='transport stop'):
        asyncio.run(qualifier.qualify_context(environment, task_dir=task, output=output,
            manifest=manifest, profile=SOURCE384_PROFILE, intent_requirement_contract=contract))
    calls = environment.upload_file.await_args_list
    assert json.loads(Path(calls[0].args[0]).read_bytes()) == contract
    assert calls[0].args[1].endswith('/source384-intent-requirements.json')
    import shlex, ast
    arguments = shlex.split(environment.exec.await_args.kwargs['command'])
    assert arguments[-1] == calls[0].args[1]
    tree = ast.parse(arguments[arguments.index('-c') + 1])
    prepared_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'prepare')
    # The executed probe supplies the uploaded path to the actual preparation owner.
    assert any(keyword.arg is None and 'intent_requirement_contract' in ast.unparse(keyword.value)
        and 'sys.argv[1]' in ast.unparse(keyword.value) for keyword in prepared_call.keywords)


@pytest.mark.parametrize('mutation', ['source-text', 'missing', 'selector'])
def test_qualifier_invalid_review_refuses_before_upload(tmp_path, selected, header, mutation):
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualifier
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_PROFILE
    contract, _ = header
    manifest = build(tmp_path, selected)
    task = tmp_path / 'task'; task.mkdir()
    (task / 'instruction.md').write_text('Changed instruction.' if mutation == 'source-text' else 'Repair unsafe headers.')
    output = tmp_path / 'output'; output.mkdir()
    if mutation == 'missing': contract = None
    if mutation == 'selector': contract['source_applicability']['review_ref'] = 'review:changed@1'
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock())
    with pytest.raises(ValueError):
        asyncio.run(qualifier.qualify_context(environment, task_dir=task, output=output,
            manifest=manifest, profile=SOURCE384_PROFILE, intent_requirement_contract=contract))
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()


@pytest.mark.parametrize('mutation', [None, 'intent', 'header', 'missing-header-owner'])
def test_qualified_header_receipt_requires_archive_owner_and_exact_intent(tmp_path, selected, header, mutation):
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualifier
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_PROFILE
    contract, _ = header
    manifest = build(tmp_path, selected)
    task = tmp_path / 'task'; task.mkdir(); (task / 'instruction.md').write_text('Repair unsafe headers.')
    output = tmp_path / 'output'; output.mkdir()
    producer = dict(consumer='a'*64, config='b'*64, byte_reader='c'*64, source_owners={})
    for key, name in [('consumer', 'source384_repository_context'), ('config', 'source384_config'),
                      ('byte_reader', 'security_autoencoder_advisor')]:
        manifest['files'].append(dict(path='source/ipfs_accelerate_py/agent_supervisor/runtime/'+name+'.py', sha256=producer[key]))
    if mutation != 'missing-header-owner':
        manifest['files'].append(dict(path='source/ipfs_accelerate_py/agent_supervisor/runtime/header_intent_applicability.py', sha256='d'*64))
    inference = json.dumps({'report': {'key': {'authored_transport_receipt': True}}}).encode()
    result = dict(qualified=True, provider_calls=0, official_verifier_executed=False, benchmark_result=False,
        checkpoint_sha256=manifest['source384']['config']['checkpoint_sha256'],
        config_sha256=manifest['source384']['config_sha256'], producer=producer,
        inference_sha256=hashlib.sha256(inference).hexdigest(),
        native_inference_key=json.loads(inference)['report']['key'],
        intent_requirement_contract_cid=cid_for_dag_json(contract), header_consumer_sha256='d'*64)
    if mutation == 'intent': result['intent_requirement_contract_cid'] = 'different'
    if mutation == 'header': result['header_consumer_sha256'] = 'e'*64
    async def download(source, destination):
        destination.write_bytes(json.dumps(result).encode() if source == qualifier.RESULT_PATH else inference)
    environment = SimpleNamespace(upload_file=AsyncMock(), download_file=download,
        exec=AsyncMock(return_value=SimpleNamespace(stdout='', stderr='', return_code=0)))
    call = qualifier.qualify_context(environment, task_dir=task, output=output, manifest=manifest,
        profile=SOURCE384_PROFILE, intent_requirement_contract=contract)
    if mutation:
        with pytest.raises(ValueError): asyncio.run(call)
    else:
        assert asyncio.run(call) == result
