"""Native parser/config/supervisor/daemon namespace regression; no DB opens."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import configured_board_scheduler as scheduler
from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as master
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import DatabaseProgramConfig
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as supervisor
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as bridge

C=Path(supervisor.__file__).resolve().parents[3]
CONFIG=C/'docs/architecture/legal_ir_grok_parallel_20261016/scheduler_config.json'
PACKET=CONFIG.with_name('execution-packet.json')
MISMATCHED_BRANCH='agent/namespace-regression-unrelated-branch'


def _board():
    board=scheduler.load_configured_board(CONFIG,repo_root=C)
    assert board.database_program.authority_mode=='quack'
    return replace(board,merge_target_branch=MISMATCHED_BRANCH)


def _supervisor(board,tmp_path,*,omit_namespace=False,defaults=False,command=''):
    configured_common=list(scheduler.configured_board_common_args(board,implement=True))
    if omit_namespace:
        index=configured_common.index('--board-namespace');del configured_common[index:index+2]
    argv=['--repo-root',str(tmp_path),'--implementation-supervisor-command',command]
    if defaults:
        argv.append('--implementation-supervisor-defaults')
    argv.extend('--common-arg='+item for item in configured_common)
    master_args=master.build_arg_parser().parse_args(argv)
    common=master.common_args_from_parsed_args(master_args)
    parsed=supervisor.parse_args([*common,'--state-dir',str(tmp_path/'private-state'),
                                  '--state-prefix','namespace_fixture'])
    config=supervisor.supervisor_config_from_args(parsed,repo_root=tmp_path)
    native=supervisor.PortalImplementationSupervisor(config)
    # This constructor returns before any store access for database authority;
    # command construction/parsing do not run the daemon or mint a credential.
    command=native._build_daemon_command()
    daemon_args=daemon.parse_args(command[command.index('--interval'):])
    return common,config,native,daemon_args


def test_actual_configured_quack_namespace_survives_unrelated_merge_branch(tmp_path):
    board=_board()
    packet=json.loads(PACKET.read_text())
    command=packet['implementation_command_text_template']
    assert str(CONFIG.with_name('planning_worker.py')) in command
    common,config,native,daemon_args=_supervisor(board,tmp_path,command=command)
    assert common[common.index('--board-namespace')+1]==board.board_namespace
    assert config.board_namespace==native.board_namespace==daemon_args.board_namespace==board.board_namespace
    assert config.merge_target_branch==MISMATCHED_BRANCH
    assert config.database_program.authority_mode==daemon_args.authority_mode=='quack'
    assert config.implementation_command==daemon_args.implementation_command==command
    assert common.count('--implementation-command')==1
    # Reproduce the observed original native guard, then prove the propagated
    # namespace renders the same canonical task without weakening that guard.
    attempt=SimpleNamespace(task_cid='fixture-task',task_alias='LIR-9001',attempt_id='fixture-attempt',
        claim_id='fixture-claim',lease_id='fixture-lease',attempt_number=1,
        owner_session_id='fixture-owner',fencing_token=1,fence_epoch=1)
    record=SimpleNamespace(task_alias='LIR-9001',priority='P2',body={
        'objective':'Namespace-only fixture', 'board_namespace':board.board_namespace})
    with pytest.raises(bridge.DatabasePortalBridgeError,match='board namespace conflicts'):
        bridge.DatabasePortalExecutionBridge._render_projection_seed(attempt,record,
            board_namespace=MISMATCHED_BRANCH.rsplit('/',1)[-1])
    rendered=bridge.DatabasePortalExecutionBridge._render_projection_seed(attempt,record,
        board_namespace=native.board_namespace)
    assert '- Board namespace: '+board.board_namespace in rendered
    assert not list(tmp_path.rglob('*.duckdb'))


def test_omitted_explicit_namespace_preserves_existing_branch_inference(tmp_path):
    board=_board()
    _common,config,native,daemon_args=_supervisor(board,tmp_path,omit_namespace=True)
    assert config.board_namespace==''
    assert native.board_namespace==daemon_args.board_namespace==MISMATCHED_BRANCH.rsplit('/',1)[-1]
    assert not list(tmp_path.rglob('*.duckdb'))


def test_legacy_configured_board_keeps_existing_namespace_inference_inputs():
    board=_board()
    legacy=DatabaseProgramConfig(authority_mode='legacy_markdown',task_source_kind='legacy-markdown',
                                 explicit_legacy=True)
    common=scheduler.configured_board_common_args(replace(board,database_program=legacy),implement=False)
    assert '--board-namespace' not in common
    assert common[common.index('--merge-target-branch')+1]==MISMATCHED_BRANCH


@pytest.mark.parametrize('defaults',[False,True])
def test_explicit_worker_command_is_independent_of_standard_defaults(tmp_path,defaults):
    command='/usr/bin/python3 -B protected-planning-worker.py --tools none'
    common,config,_native,daemon_args=_supervisor(_board(),tmp_path,
                                                 defaults=defaults,command=command)
    assert common.count('--implementation-command')==1
    assert config.implementation_command==daemon_args.implementation_command==command


def test_common_arg_worker_override_keeps_existing_last_value_precedence():
    args=master.build_arg_parser().parse_args([
        '--implementation-supervisor-command','selected-first',
        '--common-arg=--implementation-command','--common-arg=selected-last'])
    effective=master.common_args_from_parsed_args(args)
    parsed=supervisor.parse_args(effective)
    assert parsed.implementation_command=='selected-last'
