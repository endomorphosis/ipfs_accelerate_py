"""Sealed projections preserve the exact execution alias and board identity."""
from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeError, DatabasePortalExecutionBridge
from test.api.test_agent_supervisor_database_portal_bridge import _TaskSource, _attempt, _record
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_text


def test_execution_board_is_sealed_and_survives_private_projection_filename(tmp_path):
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()), attempt_root=tmp_path / 'attempts',
        portal_factory=lambda *_: None, board_namespace='intent',
    )
    paths, binding = bridge._ensure_attempt_projection(_attempt(), _record())
    projection = bridge._verify_projection(paths, binding)
    assert projection.count('- Board namespace: intent') == 1
    task = parse_task_text(projection, path=paths.task_projection, task_header_prefix='## LGSWF-004')[0]
    assert task.board_namespace == 'intent'
    assert task.board_namespace != paths.task_projection.name
    # Independent owner reconstruction uses the same instance-bound renderer.
    assert binding == bridge._binding(_attempt(), _record(), bridge._render_projection(_attempt(), _record()))
    paths.task_projection.write_text(projection.replace('- Board namespace: intent', '- Board namespace: foreign'))
    with pytest.raises(DatabasePortalBridgeError, match='outside its mutable status'):
        bridge._verify_projection(paths, binding)
    paths.task_projection.write_text(projection)
    bridge.board_namespace = 'foreign'
    with pytest.raises(DatabasePortalBridgeError, match='binding changed across resume'):
        bridge._ensure_attempt_projection(_attempt(), _record())


@pytest.mark.parametrize('key', ['board_namespace', 'Board Namespace', ' board namespace ', 'board__namespace', 'board\t namespace'])
@pytest.mark.parametrize('value', ['intent', 'foreign', '', None, 'intent\n'])
def test_execution_namespace_rejects_conflicting_body_metadata(tmp_path, key, value):
    record = _record()
    record.body[key] = value
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record), attempt_root=tmp_path / 'attempts',
        portal_factory=lambda *_: None, board_namespace='intent',
    )
    if value != 'intent':
        with pytest.raises(DatabasePortalBridgeError, match='board namespace conflicts'):
            bridge._ensure_attempt_projection(_attempt(), record)
        assert not bridge._paths(_attempt()).binding.exists()
    else:
        projection = bridge._render_projection(_attempt(), record)
        assert projection.lower().count('- board namespace:') == 1


@pytest.mark.parametrize('declared_namespace', ['', 'legacy-board'])
def test_unspecified_execution_namespace_preserves_legacy_projection(tmp_path, declared_namespace):
    record = _record()
    if declared_namespace:
        record.body['board_namespace'] = declared_namespace
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record), attempt_root=tmp_path / 'attempts',
        portal_factory=lambda *_: None,
    )
    paths, binding = bridge._ensure_attempt_projection(_attempt(), record)
    task = parse_task_text(bridge._verify_projection(paths, binding), path=paths.task_projection, task_header_prefix='## LGSWF-004')[0]
    assert task.board_namespace == (declared_namespace or paths.task_projection.name)


@pytest.mark.parametrize('prefix', ['## ', '## OTHER-', '## LGSWF-'])
def test_prior_identity_and_outputs_parse_exact_sealed_alias(tmp_path, prefix):
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()), attempt_root=tmp_path / 'attempts',
        portal_factory=lambda *_: None, task_header_prefix=prefix,
        board_namespace='sealed-alias-qualification',
    )
    paths, binding = bridge._ensure_attempt_projection(_attempt(), _record())
    identity = bridge._prior_projection_identity(paths, binding)
    assert identity['task_id'] == _record().task_alias
    assert identity['canonical_task_cid'] and identity['canonical_task_key']
    assert bridge._prior_declared_output_paths(paths, binding) == ('inventory/result.json',)


@pytest.mark.parametrize('mutation', ['foreign-alias', 'extra-task', 'alias-prefix-neighbor'])
def test_rehashed_projection_cannot_admit_foreign_or_extra_task(tmp_path, mutation):
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()), attempt_root=tmp_path / 'attempts',
        portal_factory=lambda *_: None, task_header_prefix='## ',
    )
    paths, binding = bridge._ensure_attempt_projection(_attempt(), _record())
    original = paths.task_projection.read_text()
    if mutation == 'extra-task':
        changed = original + '\n## FOREIGN-001 Unadmitted task\n- Status: ready\n'
    else:
        replacement = 'FOREIGN-001' if mutation == 'foreign-alias' else 'LGSWF-004-EXTRA'
        changed = original.replace('## LGSWF-004 ', '## ' + replacement + ' ', 1)
    assert changed != original
    paths.task_projection.write_text(changed)
    # Even a recomputed observation digest cannot replace the exact claimed
    # alias or add a second task. Actual sealed bindings have further checks.
    changed_binding = {**binding, 'projection_immutable_digest': module._projection_immutable_digest(changed)}
    for reader in (bridge._prior_projection_identity, bridge._prior_declared_output_paths):
        with pytest.raises(DatabasePortalBridgeError, match='exactly the claimed task'):
            reader(paths, changed_binding)
