"""Actual authored subprocess frames through capture, router, and metric export."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_native_usage import grok_usage_receipt, grok_outcome_receipt
from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(llm_router, 'find_grok_cli', lambda: '/authored-only/grok')
    monkeypatch.setattr(llm_router, '_cli_available', lambda _: True)
    set_last_cli_observation('grok_cli', {})
    return llm_router._get_grok_cli_provider()


def command(tmp_path, frames, *, suffix=''):
    script = tmp_path / 'authored_cli.py'
    script.write_text('import sys,time\nfrom pathlib import Path\n'
        "assert sys.argv[sys.argv.index('--output-format')+1]=='streaming-json'\n"
        "p=Path(sys.argv[sys.argv.index('--prompt-file')+1])\n"
        "assert p.read_text()=='AUTHORED_ONLY_PROMPT'\n"
        f"Path({str(tmp_path/'prompt_path')!r}).write_text(str(p))\n"
        + '\n'.join(f'print({json.dumps(frame,ensure_ascii=False)!r},flush=True)' for frame in frames)
        + '\n' + suffix)
    return [sys.executable,str(script)]


def end():
    return dict(type='end',stopReason='end_turn',num_turns=2,usage_is_incomplete=False,
        usage=dict(input_tokens=12,output_tokens=4))


def closed_metric():
    obs=get_last_cli_observation('grok_cli')
    row=dict(invocation_id='authored-invocation',phase='coding',provider='grok_cli',
        native_rollout_usage=grok_usage_receipt(obs),native_provider_outcome=grok_outcome_receipt(obs),
        native_grok_progress=obs['native_grok_progress'])
    metric,=_invocations(dict(provider_invocations=[row]))
    return obs,metric


def assert_closed(metric):
    encoded=json.dumps(metric,ensure_ascii=False)
    for text in ('AUTHORED_ONLY','rawInput','rawOutput','toolCallId'):
        assert text not in encoded
    tools=metric['native_grok_progress']['tool_event_observations']
    assert tools['provider_request_bytes'] is tools['history_resend_bytes'] is None
    assert tools['request_history_complete'] is False
    assert tools['event_origin_authenticated'] is False
    assert tools['provider_tokens'] is None and tools['token_savings_claimed'] is False
    return tools


def test_repeated_payloads_reach_metric_without_inventing_resend_bytes(provider,tmp_path):
    payload={'line':'AUTHORED_ONLY_OUTPUT_λ','empty':[]}
    frames=[dict(type='tool_call',toolCallId='AUTHORED_ONLY_CALL_A',toolName='read_file',
        title='AUTHORED_ONLY_TITLE',kind='read',status='in_progress',
        rawInput={'path':'AUTHORED_ONLY_PATH'},content=[],locations=[]),
        dict(type='tool_call_update',toolCallId='AUTHORED_ONLY_CALL_A',status='completed',rawOutput=payload),
        dict(type='tool_call_update',toolCallId='AUTHORED_ONLY_CALL_A',status='completed',rawOutput=payload),
        dict(type='tool_call',toolCallId='AUTHORED_ONLY_CALL_B',toolName='read_file',status='in_progress',
             rawInput={'path':'AUTHORED_ONLY_PATH'}),
        dict(type='tool_call_update',toolCallId='AUTHORED_ONLY_CALL_B',status='completed',rawOutput=payload),
        dict(type='text',data='AUTHORED_ONLY_FINAL'),end()]
    snapshots=[]
    result=provider.generate('AUTHORED_ONLY_PROMPT',model_name='authored-model',timeout=3,
        grok_cli_cmd=command(tmp_path,frames),grok_stream_progress=True,grok_tool_event_census=True,
        grok_progress_callback=snapshots.append)
    assert result=='AUTHORED_ONLY_FINAL'
    obs,metric=closed_metric();tools=assert_closed(metric)
    assert metric['native_grok_progress']['schema']=='native-grok-stream-progress@4'
    assert tools['observed_events']==tools['retained_events']==5
    assert tools['omitted_events']==0 and tools['call_identities']==2
    output=tools['payloads']['output'];size=len(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
    assert output['measured_fields']==3 and output['canonical_json_bytes']==3*size
    assert output['repeated_events']==2 and output['repeated_bytes']==2*size
    assert output['same_call_repeated_events']==output['cross_call_repeated_events']==1
    assert metric['known_token_subtotals']['input_tokens']==12 and metric['known_token_subtotals']['output_tokens']==4
    assert all(value is None for value in metric['tokens'].values())
    assert metric['native_usage_complete_observed'] is None
    assert obs['native_grok_progress']['process_exit_observed'] is True
    assert not Path((tmp_path/'prompt_path').read_text()).exists()
    assert snapshots and all(s['schema']=='native-grok-stream-progress@1' for s in snapshots)
    assert all('tool_event_observations' not in s for s in snapshots)


@pytest.mark.parametrize('value,state',[(None,'null'),({},'empty'),('','empty'),([],'empty')])
def test_empty_and_null_are_measured_separately_from_missing(provider,tmp_path,value,state):
    frames=[dict(type='tool_call',toolCallId='AUTHORED_ONLY_CALL',toolName='read_file',status='in_progress'),
        dict(type='tool_call_update',toolCallId='AUTHORED_ONLY_CALL',status='completed',rawOutput=value),
        dict(type='text',data='AUTHORED_ONLY_FINAL'),end()]
    provider.generate('AUTHORED_ONLY_PROMPT',model_name='authored-model',timeout=3,
        grok_cli_cmd=command(tmp_path,frames),grok_stream_progress=True,grok_tool_event_census=True)
    _,metric=closed_metric();tools=assert_closed(metric)
    assert tools['rows'][0]['output']['state']=='missing'
    assert tools['rows'][1]['output']['state']==state
    assert tools['payloads']['output']['measured_fields']==1


def test_timeout_retains_closed_counts_and_unknown_request_history(provider,tmp_path):
    frames=[dict(type='tool_call',toolCallId='AUTHORED_ONLY_CALL',toolName='read_file',status='in_progress',
        rawInput={'path':'AUTHORED_ONLY_PATH'})]
    with pytest.raises(subprocess.TimeoutExpired):
        provider.generate('AUTHORED_ONLY_PROMPT',model_name='authored-model',timeout=.4,
            grok_cli_cmd=command(tmp_path,frames,suffix='time.sleep(30)\n'),
            grok_stream_progress=True,grok_tool_event_census=True)
    _,metric=closed_metric();tools=assert_closed(metric)
    assert tools['observed_events']==1 and tools['observation_complete'] is False
    assert metric['native_grok_progress']['timed_out'] is True
    assert all(value is None for value in metric['tokens'].values())
    assert not Path((tmp_path/'prompt_path').read_text()).exists()


def test_opt_out_preserves_historical_progress_and_absent_tool_measurements(provider,tmp_path):
    frames=[dict(type='text',data='AUTHORED_ONLY_FINAL'),end()]
    provider.generate('AUTHORED_ONLY_PROMPT',model_name='authored-model',timeout=3,
        grok_cli_cmd=command(tmp_path,frames),grok_stream_progress=True)
    _,metric=closed_metric()
    assert metric['native_grok_progress']['schema']=='native-grok-stream-progress@3'
    assert 'tool_event_observations' not in metric['native_grok_progress']
    assert metric['known_token_subtotals']['input_tokens']==12
    assert all(value is None for value in metric['tokens'].values())


def test_overflow_is_bounded_and_does_not_change_provider_result(provider,tmp_path):
    from ipfs_accelerate_py.cli_runtime.grok_tool_events import MAX_TOOL_EVENTS
    frames=[dict(type='tool_call',toolCallId='AUTHORED_ONLY_CALL_'+str(i),toolName='read_file',
        status='in_progress',rawInput={'path':'AUTHORED_ONLY_PATH'}) for i in range(MAX_TOOL_EVENTS+2)]
    frames.extend([dict(type='text',data='AUTHORED_ONLY_FINAL'),end()])
    assert provider.generate('AUTHORED_ONLY_PROMPT',model_name='authored-model',timeout=3,
        grok_cli_cmd=command(tmp_path,frames),grok_stream_progress=True,
        grok_tool_event_census=True)=='AUTHORED_ONLY_FINAL'
    _,metric=closed_metric();tools=assert_closed(metric)
    assert tools['observed_events']==MAX_TOOL_EVENTS+2
    assert tools['retained_events']==MAX_TOOL_EVENTS and tools['omitted_events']==2
    assert len(tools['rows'])==MAX_TOOL_EVENTS and tools['observation_complete'] is False
    assert tools['incompleteness']['row_limit_exceeded'] is True


def test_unknown_tool_payload_fields_retain_narrow_incomplete_scope(provider,tmp_path):
    frames=[dict(type='tool_call',toolCallId='AUTHORED_ONLY_CALL',toolName='AUTHORED_ONLY_TOOL',
        status='in_progress',rawInput={'path':'AUTHORED_ONLY_PATH'},futureField='AUTHORED_ONLY_EXTRA'),
        dict(type='text',data='AUTHORED_ONLY_FINAL'),end()]
    assert provider.generate('AUTHORED_ONLY_PROMPT',model_name='authored-model',timeout=3,
        grok_cli_cmd=command(tmp_path,frames),grok_stream_progress=True,
        grok_tool_event_census=True)=='AUTHORED_ONLY_FINAL'
    _,metric=closed_metric();tools=assert_closed(metric)
    assert tools['rows'][0]['tool']=='unknown'
    assert tools['rows'][0]['unknown_field_count']==1
    assert tools['observation_complete'] is False
