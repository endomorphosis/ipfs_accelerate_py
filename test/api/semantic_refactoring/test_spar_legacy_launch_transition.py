"""Real Git source continuity and real retained DuckDB/Quack cursor migration."""
from dataclasses import replace
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor import spar_merge_owner as role
from scripts.ops.agent_supervisor import spar_merge_owner_handoff as handoff
from scripts.ops.agent_supervisor import spar_legacy_launch_transition as transition
from test.api.semantic_refactoring.test_launch_source_amendment_task_source import _fixture
from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import (
    preserved as _preserved, add_preserved_imports, start, attach_recovery,
)


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _seal(root):
    value = {'control_file_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                    for name in ('taskboard.md', 'objectives.md', 'plan.md', 'validator.py')},
             'bootstrap_runtime_file_sha256': {'runtime.py': hashlib.sha256((root/'runtime.py').read_bytes()).hexdigest()}}
    value['seal_cid'] = transition._native_cid(value)
    _write_json(root/'dependencies.json', value)
    return value


@pytest.fixture
def preserved(tmp_path):
    return _preserved.__wrapped__(tmp_path)


@pytest.fixture
def source_transition(tmp_path, preserved):
    root = tmp_path/'repo'
    root.mkdir()
    _git(root, 'init', '-q', '-b', 'main')
    _git(root, 'config', 'user.name', 'SPAR source qualification')
    _git(root, 'config', 'user.email', 'spar-source@example.invalid')
    (root/'.gitignore').write_text('/state/\n')
    for name in ('taskboard.md', 'objectives.md', 'plan.md', 'validator.py'):
        (root/name).write_text('immutable '+name+'\n')
    (root/'runtime.py').write_text('VALUE=1\n')
    old_seal = _seal(root)
    old_config = {'dependency_seal_path':'dependencies.json', 'dependency_seal_cid':old_seal['seal_cid'],
                  'merge_target_branch':'main', 'taskboard_path':'taskboard.md', 'objectives_path':'objectives.md',
                  'plan_path':'plan.md', 'validator_path':'validator.py',
                  'runtime_paths':{'state':'state'}, 'max_lanes':1}
    _write_json(root/'config.json', old_config)
    old_config_raw = (root/'config.json').read_bytes()
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'Original sealed bootstrap')
    old_head = _git(root, 'rev-parse', 'HEAD')
    old_tree = _git(root, 'rev-parse', 'HEAD^{tree}')
    _, _, example, *_ = _fixture()
    bootstrap = {'source_head':old_head, 'repository_tree_id':old_tree,
                 'plan_root_cid':example.bootstrap_plan_root_cid,
                 'source_identities':{name:transition._cid_bytes((root/old_config[name+'_path']).read_bytes())
                                      for name in ('taskboard','objectives','plan','validator')}}
    bootstrap['source_identities']['config'] = transition._cid_bytes(old_config_raw)
    bootstrap['bootstrap_receipt_id'] = transition._native_cid(bootstrap)
    paths = {'bootstrap_receipt':root/'state/bootstrap.json'}
    _write_json(paths['bootstrap_receipt'], bootstrap)
    (root/'runtime.py').write_text('VALUE=2\n')
    new_seal = _seal(root)
    current_config = {**old_config, 'dependency_seal_cid':new_seal['seal_cid']}
    _write_json(root/'config.json', current_config)
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'Current runtime seal without bootstrap rewrite')
    head = _git(root, 'rev-parse', 'HEAD')
    tree = _git(root, 'rev-parse', 'HEAD^{tree}')
    forest = {'source_head':head,'nested_repositories':[], 'cross_repository_writes':False}
    forest['source_forest_root'] = transition._native_cid(forest)
    receipt = {'schema':example.launch_source_forest_receipt['schema'], 'source_head':head,'repository_tree':tree,
               'source_forest_root':forest['source_forest_root'],'source_forest':forest}
    receipt['receipt_id'] = transition._native_cid(receipt)
    amendment = replace(example, board_namespace='spar', bootstrap_receipt_id=bootstrap['bootstrap_receipt_id'],
        bootstrap_source_head=old_head, bootstrap_repository_tree_id=old_tree,
        bootstrap_config_cid=transition._cid_bytes(old_config_raw),
        launch_config_cid=transition._cid_bytes((root/'config.json').read_bytes()),
        launch_source_head=head, launch_repository_tree_id=tree,
        launch_source_forest_receipt=receipt, launch_source_forest_receipt_id=receipt['receipt_id'],
        launch_source_forest_root=forest['source_forest_root'], dependency_seal_cid=new_seal['seal_cid'],
        **{'immutable_'+name+'_cid':bootstrap['source_identities'][name]
           for name in ('taskboard','objectives','plan','validator')}, amendment_id='')
    board = SimpleNamespace(repo_root=root, config_path=root/'config.json', payload=current_config,
                            board_namespace='spar', max_lanes=1, task_prefix='SPAR-')
    board.path = lambda name: root/name
    source, manifest, pending, unknown, _ = preserved
    old_scope = {'board_namespace':'spar','plan_cid':amendment.bootstrap_plan_root_cid,
                 'config_cid':transition._cid_bytes(old_config_raw),'lane_id':'0',
                 'attempt_root':str(root/'state/lane-0/spar_lane_0_database_portal_attempts')}
    manifest = {**manifest, 'source_commit':old_head,'source_tree':old_tree,'scope_bindings':[old_scope]}
    cursor, receipts = add_preserved_imports(source, manifest)
    origin = {'manifest':manifest, 'capture':{'source':{'head':old_head,'tree':old_tree,
              'config_sha256':hashlib.sha256(old_config_raw).hexdigest()}}}
    scopes = handoff._scopes(board=board, paths=paths, amendment=amendment)
    return SimpleNamespace(root=root, board=board, paths=paths, amendment=amendment, origin=origin,
        scopes=scopes, source=source, manifest=manifest, cursor=cursor, pending=pending, unknown=unknown,
        bootstrap_bytes=paths['bootstrap_receipt'].read_bytes())


def qualify(case):
    return transition.qualify_transition(board=case.board, paths=case.paths,
        amendment=case.amendment, origin=case.origin, scopes=case.scopes)


def test_committed_current_source_preserves_original_bootstrap(source_transition):
    case = source_transition
    result = qualify(case)
    assert result.receipt['configuration_delta'] == ['dependency_seal_cid']
    assert result.scope_bindings[0]['config_cid'] == case.amendment.launch_config_cid
    assert result.receipt['old_config_cid'] != result.receipt['current_config_cid']
    assert case.paths['bootstrap_receipt'].read_bytes() == case.bootstrap_bytes
    assert _git(case.root, 'status', '--porcelain') == ''


@pytest.mark.parametrize('which', ['attempt_root','board_namespace','lane_id','plan_cid'])
def test_scope_transition_rejects_namespace_changes(source_transition, which):
    case = source_transition
    case.scopes[0][which] += 'foreign'
    with pytest.raises(role.SparMergeOwnerError, match='namespace differs'):
        qualify(case)


def test_source_transition_rejects_uncommitted_runtime(source_transition):
    (source_transition.root/'runtime.py').write_text('UNSEALED=True\n')
    with pytest.raises(role.SparMergeOwnerError, match='not clean'):
        qualify(source_transition)


def test_source_transition_rejects_rewritten_bootstrap(source_transition):
    case = source_transition
    body = json.loads(case.paths['bootstrap_receipt'].read_text())
    body['source_identities']['config'] = case.amendment.launch_config_cid
    body['bootstrap_receipt_id'] = transition._native_cid({k:v for k,v in body.items() if k!='bootstrap_receipt_id'})
    _write_json(case.paths['bootstrap_receipt'],body)
    with pytest.raises(role.SparMergeOwnerError, match='bootstrap differs'):
        qualify(case)


def test_current_scope_inherits_retained_cursor_and_restart_keeps_progress(source_transition, tmp_path):
    case = source_transition
    prepared = role.prepare_offline_clone(offline_root=case.source, destination=tmp_path/'prepared', manifest=case.manifest)
    prepared = replace(prepared, launch_transition=qualify(case))
    manifest = {**case.manifest, 'scope_bindings':case.scopes}
    server = start(prepared, tmp_path/'owner')
    client = None
    try:
        client, api = attach_recovery(server, manifest)
        initial = api.load_cursors()
        assert initial['cursors'] == case.cursor['cursors']
        changed = {**initial['cursors'], 'pending_requests':'successor-progress'}
        result = api.cas_cursors(expected_revision=initial['revision'], expected_state_cid=initial['state_cid'],
            cursors=changed, operation_id='advance-current-scope')
        assert result['cursors'] == changed
        with server._owner_transaction_lock:
            role.require_preserved(prepared.preserved_inventory, role.inventory(server._connection))
        identity = server.identity
    finally:
        if client: client.close()
        server.stop()
    server = start(prepared, tmp_path/'owner')
    try:
        client, api = attach_recovery(server, manifest)
        assert api.load_cursors()['cursors'] == changed
        assert server.identity.generation == identity.generation + 1
        assert server.identity.database_uuid == identity.database_uuid
        original, old_api = attach_recovery(server, case.manifest, consumer='original-scope-audit')
        try:
            assert old_api.load_cursors()['cursors'] == case.cursor['cursors']
        finally:
            original.close()
    finally:
        if client: client.close()
        server.stop()
    assert case.paths['bootstrap_receipt'].read_bytes() == case.bootstrap_bytes


def test_transition_refuses_resetting_an_advanced_original_scope(source_transition, tmp_path):
    case = source_transition
    prepared = role.prepare_offline_clone(offline_root=case.source, destination=tmp_path/'prepared', manifest=case.manifest)
    server = start(prepared, tmp_path/'owner')
    client = None
    try:
        client, api = attach_recovery(server, case.manifest)
        head = api.load_cursors()
        api.cas_cursors(expected_revision=head['revision'], expected_state_cid=head['state_cid'],
            cursors={**head['cursors'], 'pending_requests':'new-original-progress'}, operation_id='old-progress')
    finally:
        if client: client.close()
        server.stop()
    with pytest.raises(role.SparMergeOwnerError, match='original cursor advanced'):
        start(replace(prepared, launch_transition=qualify(case)), tmp_path/'owner')


def _advance_source(case, *, config_change=None):
    (case.root/'runtime.py').write_text('VALUE=3\n')
    seal = _seal(case.root)
    config = {**case.board.payload, 'dependency_seal_cid':seal['seal_cid'], **(config_change or {})}
    _write_json(case.root/'config.json', config)
    _git(case.root, 'add', '.')
    _git(case.root, 'commit', '-qm', 'Another current source')
    head = _git(case.root, 'rev-parse', 'HEAD')
    tree = _git(case.root, 'rev-parse', 'HEAD^{tree}')
    forest = {'source_head':head, 'nested_repositories':[], 'cross_repository_writes':False}
    forest['source_forest_root'] = transition._native_cid(forest)
    receipt = {'schema':case.amendment.launch_source_forest_receipt['schema'], 'source_head':head,
               'repository_tree':tree, 'source_forest_root':forest['source_forest_root'],'source_forest':forest}
    receipt['receipt_id'] = transition._native_cid(receipt)
    case.board.payload = config
    case.amendment = replace(case.amendment, launch_source_head=head, launch_repository_tree_id=tree,
        launch_source_forest_root=forest['source_forest_root'], launch_source_forest_receipt=receipt,
        launch_source_forest_receipt_id=receipt['receipt_id'], dependency_seal_cid=seal['seal_cid'],
        launch_config_cid=transition._cid_bytes((case.root/'config.json').read_bytes()), amendment_id='')
    case.scopes = handoff._scopes(board=case.board, paths=case.paths, amendment=case.amendment)


def test_other_configuration_change_requires_separate_admission(source_transition):
    case = source_transition
    _advance_source(case, config_change={'max_lanes':2})
    with pytest.raises(role.SparMergeOwnerError, match='only a dependency seal identity change'):
        qualify(case)


def test_changed_qualified_transition_is_revalidated_before_migration(source_transition, tmp_path):
    case = source_transition
    prepared = role.prepare_offline_clone(offline_root=case.source, destination=tmp_path/'prepared', manifest=case.manifest)
    qualified = qualify(case)
    qualified.scope_bindings[0]['attempt_root'] += '/foreign'
    with pytest.raises(role.SparMergeOwnerError, match='changed after source qualification'):
        start(replace(prepared, launch_transition=qualified), tmp_path/'owner')


def test_second_configuration_does_not_reseed_current_cursor(source_transition, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import OwnerRecoveryRuntimeError
    case = source_transition
    prepared = role.prepare_offline_clone(offline_root=case.source, destination=tmp_path/'prepared', manifest=case.manifest)
    admitted = replace(prepared, launch_transition=qualify(case))
    server = start(admitted, tmp_path/'owner')
    client = None
    try:
        client, api = attach_recovery(server, {**case.manifest, 'scope_bindings':case.scopes})
        receipt = api.get_receipt('native-launch-scope-transition:'+prepared.manifest_cid)
        assert receipt['receipt'] == admitted.launch_transition.receipt
        assert receipt['receipt']['completion_authority'] is False
    finally:
        if client: client.close()
        server.stop()
    _advance_source(case)
    with pytest.raises(OwnerRecoveryRuntimeError, match='different preserved state'):
        start(replace(prepared, launch_transition=qualify(case)), tmp_path/'owner')
