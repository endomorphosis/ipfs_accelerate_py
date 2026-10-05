"""Frozen real owner re-sign diagnostic, prior to profile/manifest join fix."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.agent_supervisor.container_coding.test_terminal_multitask_preparation import multitask_case, _prepare
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as symbolic
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import verify_owner_local_benchmark_observation
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

output = Path(__file__).parent
source = Path(local.__file__)
source_before = source.read_bytes()
(output / 'local_planning_admission-owner-red.py').write_bytes(source_before)
with TemporaryDirectory(prefix='multitask-owner-red-') as temporary:
    case = multitask_case.__wrapped__(Path(temporary))
    prepared = _prepare(case)
    payload = deepcopy(prepared['manifest']['payload'])
    contract = deepcopy(case['contract'])
    contract['symbolic_operations']['review_ref'] = 'review:substituted-owner-contract@1'
    artifact = payload['intent_requirements']
    artifact['contract_json'] = json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    artifact['contract_cid'] = cid_for_dag_json(contract)
    manifest = local._signed(payload, payload)
    local._manifest(manifest, initial=True)
    plan = symbolic.build_intent_symbolic_plan(contract, manifest=manifest)
    admission = local.admit_local_benchmark_plan(graph=plan['graph'], manifest=manifest,
        requirement_bindings=plan['requirement_bindings'])
    database = case['state'] / 'forged-native.duckdb'
    with IntentRepository(database) as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        count = len(intent.list_tasks())
    with open_existing_native_owner(database=database, checkout=case['repository'],
        state_dir=Path(temporary) / 'native-owner', repository_id=payload['repository_cid'],
        execution_routes={row.task_key: GROK_CODEX_EXECUTION_MODE for row in plan['graph'].tasks}) as owner:
        observed = verify_owner_local_benchmark_observation(server=owner.server, admission=admission)
        native_count = len(owner.source.list_tasks().tasks)
    result = {'schema':'terminal-multitask-owner-profile-binding-red@1',
        'source_path':str(source), 'source_sha256':hashlib.sha256(source_before).hexdigest(),
        'source_unchanged':source_before == source.read_bytes(),
        'profile_contract_cid':case['profile']['intent_requirement_contract_cid'],
        'substituted_contract_cid':artifact['contract_cid'], 'owner_admitted':True,
        'materialized_task_count':count, 'typed_native_task_count':native_count,
        'native_observation_replay_accepted':bool(observed),
        'completion_authority':admission['receipt']['payload']['completion_authority'],
        'provider_calls':0, 'native_owner_stopped':owner.server.status()['lifecycle'] == 'stopped'}
    (output / 'owner-binding-red.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True))
