"""Exact callback retirement using real owner methods and explicit runtime fixtures.

No signed launch, native worker, training or grant authorization is claimed.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import (
    OwnerLocalCompletionService, bind_owner_local_completion_service,
)
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import LocalPlanningError
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import QuackStateServer
from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution import (
    ADVISORY_PROFILE, FrozenFiniteRepositoryExecutionScope,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerGateway, TypedStateOwnerProtocolError,
)


def owner_fixture(tmp_path):
    gateway = TypedStateOwnerGateway.__new__(TypedStateOwnerGateway)
    gateway._transaction_lock = threading.RLock()
    gateway._grants_lock = threading.Lock()
    gateway._local_task_validation_handler = None
    gateway._local_task_validation_binding = None
    gateway._local_task_validation_active = 0
    server = QuackStateServer.__new__(QuackStateServer)
    server._identity = object()
    server._lock = gateway._transaction_lock
    server._command_gateway = gateway
    repo = tmp_path/'repository'
    repo.mkdir(); (repo/'.git').mkdir()
    return server, gateway, repo


def bind(server, repo, *, retirable=True):
    return bind_owner_local_completion_service(server=server, repo_root=repo,
        portal_attempt_root=repo.parent/'attempts', merge_queue_dir=repo.parent/'queue',
        board_namespace='fixture-board', target_branch='main', retirable=retirable)


def runtime_fixture(server, service):
    runtime = AdmittedBenchmarkRuntime.__new__(AdmittedBenchmarkRuntime)
    runtime.server, runtime.completion_service = server, service
    runtime.manifest = {'candidate_runner': {'fixture': True}}
    runtime.profile = SimpleNamespace()
    runtime._context_refresh_stop_receipt = SimpleNamespace(succeeded=True,
        data={'old_tree_fenced': True, 'isolated_worker_cleanup': {'returncode': 0, 'single_worker': True}})
    runtime.process = SimpleNamespace(snapshot=lambda _p: SimpleNamespace(members=()))
    scope = FrozenFiniteRepositoryExecutionScope.__new__(FrozenFiniteRepositoryExecutionScope)
    scope._material = json.dumps({'payload': {'profile': ADVISORY_PROFILE}})
    scope._runtime, scope._spawned = runtime, False
    runtime.finite_execution_scope = scope
    runtime.observed = []
    runtime._record = lambda name, value: runtime.observed.append((name, value))
    runtime.coordinator = SimpleNamespace(release=lambda *_a, **_k: runtime.observed.append(('release', {})),
        close=lambda: runtime.observed.append(('coordinator-close', {})))
    runtime.lease = SimpleNamespace(fencing_token='fixture', fence_epoch=1)
    runtime._bootstrap_stop = threading.Event()
    runtime._listener = SimpleNamespace(close=lambda: None)
    runtime._bootstrap_thread = SimpleNamespace(join=lambda **_kw: None)
    return runtime


def test_successful_native_methods_retire_runtime_callback_and_allow_fresh_rebind(tmp_path):
    server, gateway, repo = owner_fixture(tmp_path)
    for _ in range(3):
        service = bind(server, repo)
        runtime = runtime_fixture(server, service)
        assert type(service) is OwnerLocalCompletionService and not service.retired
        assert dict(service) == {'schema':'supervisor-local-owner-validation-service@1',
            'bound':True, 'operation':'local.task.validation.run', 'completion_authority':False}
        runtime.close()
        assert service.retired and gateway._local_task_validation_handler is None
        assert gateway._local_task_validation_binding is None and gateway._local_task_validation_active == 0
        assert [x[0] for x in runtime.observed] == ['completion-service-close','release','coordinator-close']


@pytest.mark.parametrize('missing', ['stop','fence','cleanup','single_worker','members'])
def test_close_without_genuine_cleanup_observation_retains_handler(tmp_path, missing):
    server, gateway, repo = owner_fixture(tmp_path)
    service = bind(server, repo); runtime = runtime_fixture(server, service)
    receipt = runtime._context_refresh_stop_receipt
    if missing == 'stop': runtime._context_refresh_stop_receipt = None
    elif missing == 'fence': receipt.data['old_tree_fenced'] = False
    elif missing == 'cleanup': receipt.data['isolated_worker_cleanup']['returncode'] = 1
    elif missing == 'single_worker': receipt.data['isolated_worker_cleanup']['single_worker'] = False
    else: runtime.process.snapshot = lambda _p: SimpleNamespace(members=(object(),))
    with pytest.raises(LocalPlanningError, match='exact native STOP'):
        runtime.close()
    assert not service.retired and gateway._local_task_validation_handler is service._handler
    assert not runtime.observed


def test_foreign_handler_and_token_are_never_removed(tmp_path):
    server, gateway, repo = owner_fixture(tmp_path)
    service = bind(server, repo); runtime = runtime_fixture(server, service)
    assert not gateway.unbind_local_task_validation_handler(service._handler, object())
    assert not gateway.unbind_local_task_validation_handler(lambda: None, service._binding)
    foreign = lambda: None
    gateway._local_task_validation_handler = foreign
    with pytest.raises(LocalPlanningError, match='foreign handler'):
        runtime.close()
    assert gateway._local_task_validation_handler is foreign and not service.retired
    assert not runtime.observed


def test_active_callback_refuses_retirement_then_detaches_after_return(tmp_path):
    server, gateway, _repo = owner_fixture(tmp_path)
    started, finish = threading.Event(), threading.Event()
    def callback():
        started.set(); assert finish.wait(2); return 'actual callback result'
    binding = gateway.bind_local_task_validation_handler(callback, retirable=True)
    results=[]
    thread=threading.Thread(target=lambda: results.append(gateway._invoke_local_task_validation_handler()))
    thread.start(); assert started.wait(1)
    try:
        with pytest.raises(TypedStateOwnerProtocolError, match='custody is active'):
            gateway.unbind_local_task_validation_handler(callback, binding)
        assert gateway._local_task_validation_handler is callback
    finally:
        finish.set();thread.join(timeout=2)
    assert not thread.is_alive() and results == ['actual callback result']
    assert gateway.unbind_local_task_validation_handler(callback, binding)
    with pytest.raises(TypedStateOwnerProtocolError, match='unavailable'):
        gateway._invoke_local_task_validation_handler()


def test_same_thread_callback_cannot_retire_itself_reentrantly(tmp_path):
    _server, gateway, _repo = owner_fixture(tmp_path)
    def callback():
        with pytest.raises(TypedStateOwnerProtocolError, match='callback is active'):
            gateway.unbind_local_task_validation_handler(callback, binding)
    binding=gateway.bind_local_task_validation_handler(callback, retirable=True)
    gateway._invoke_local_task_validation_handler()
    assert gateway.unbind_local_task_validation_handler(callback, binding)


def test_legacy_binding_remains_exact_dict_and_duplicate_bind_refuses(tmp_path):
    server, gateway, repo = owner_fixture(tmp_path)
    service=bind(server, repo, retirable=False)
    assert type(service) is dict and not hasattr(service,'close')
    assert gateway._local_task_validation_binding is None
    assert not gateway.unbind_local_task_validation_handler(gateway._local_task_validation_handler, None)
    with pytest.raises(TypedStateOwnerProtocolError, match='already bound'):
        bind(server, repo)
    assert gateway._local_task_validation_handler is not None
    with pytest.raises(LocalPlanningError, match='exact boolean'):
        bind(server, repo, retirable='yes')


def test_already_locked_rpc_selection_supports_non_reentrant_owner_locks(tmp_path):
    _server, gateway, _repo = owner_fixture(tmp_path)
    gateway._transaction_lock = threading.Lock()
    callback = lambda: 'actual result'
    binding = gateway.bind_local_task_validation_handler(callback, retirable=True)
    with gateway._transaction_lock:
        assert gateway._invoke_local_task_validation_handler_locked() == 'actual result'
    assert gateway.unbind_local_task_validation_handler(callback, binding)
