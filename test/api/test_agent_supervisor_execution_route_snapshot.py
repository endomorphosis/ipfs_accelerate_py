"""Launch evidence must use the policy's generation-stable typed projection."""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskRecord, TaskSourceConflictError, TaskSourceIntegrityError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)


class Owner:
    def __init__(self, generations):
        self.generations = iter(generations)
        self.reads = []
        self.revision = 0

    def load_generation(self):
        self.revision = next(self.generations)
        return SimpleNamespace(content_id=f'generation:{self.revision}', revision=self.revision)

    def execute(self, operation):
        self.reads.append(operation)
        assert operation == 'executor_control_snapshot'
        return [{
            'goal_count': 1, 'task_count': 1, 'dependency_count': 0,
            'objective_count': 1, 'plan_count': 1,
            'event_watermark': self.revision * 10,
            'goals_json': '[]', 'plans_json': '[]', 'tasks_json': '[]',
        }]


def source(generations):
    owner = Owner(generations)
    adapter = object.__new__(TypedDatabaseTaskSource)
    adapter._client = owner
    def records(*, expected_count):
        assert expected_count == 1
        return ((TaskRecord(
            task_cid='task:one', task_alias='ONE', goal_cid='goal:one',
            plan_cid='plan:one', ordinal=1, status='ready', revision=owner.revision,
        ), {'repository_tree_id': 'tree:one'}),)
    adapter._all_records = records
    return adapter, owner


def test_launch_snapshot_and_policy_share_one_stable_owner_read():
    adapter, owner = source([7, 7])
    snapshot, policy = adapter.seal_execution_route_snapshot({'ONE': 'grok-codex'})
    assert snapshot.revision == policy.source_revision == 7
    assert snapshot.projection_cid == policy.source_projection_cid
    assert snapshot.event_cursor == 70
    assert policy.entries[0].task_revision == 7
    assert owner.reads == ['executor_control_snapshot']


def test_generation_change_retries_entire_pair_without_mixing_observations():
    adapter, owner = source([7, 8, 8, 8])
    snapshot, policy = adapter.seal_execution_route_snapshot({'ONE': 'grok-codex'})
    assert snapshot.revision == policy.source_revision == 8
    assert snapshot.event_cursor == 80
    assert policy.entries[0].task_revision == 8
    assert snapshot.projection_cid == policy.source_projection_cid
    assert len(owner.reads) == 2


def test_continuous_generation_drift_exhausts_existing_bound():
    adapter, owner = source(range(1, 9))
    with pytest.raises(TaskSourceConflictError):
        adapter.seal_execution_route_snapshot({'ONE': 'grok-codex'})
    assert len(owner.reads) == 4


@pytest.mark.parametrize('modes', [{}, {'OTHER': 'grok-codex'}, {'ONE': 'unknown'}])
def test_exact_population_and_execution_modes_remain_required(modes):
    adapter, _owner = source([7, 7])
    with pytest.raises(TaskSourceIntegrityError):
        adapter.seal_execution_route_snapshot(modes)


def test_owner_error_propagates_without_retry_or_local_fallback():
    adapter, owner = source([7])
    def denied(_operation):
        raise PermissionError('scope denied')
    owner.execute = denied
    with pytest.raises(PermissionError, match='scope denied'):
        adapter.seal_execution_route_snapshot({'ONE': 'grok-codex'})


def test_existing_policy_api_preserves_exact_identity():
    old, _ = source([7, 7])
    paired, _ = source([7, 7])
    assert old.seal_execution_route_policy({'ONE': 'grok-codex'}) == paired.seal_execution_route_snapshot({'ONE': 'grok-codex'})[1]
