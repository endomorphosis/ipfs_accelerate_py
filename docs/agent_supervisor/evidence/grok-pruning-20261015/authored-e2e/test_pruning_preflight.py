"""Authored native-inspect subprocesses; no Docker, credentials or model calls.

The private capture helper receives executable authored scripts. Its stdout is
then checked by the production pure inspection validator before a dispatch
sentinel. Source-checkout tests separately cover the public fixed-binary gate,
file drift, and actual router dispatch connection.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest


@pytest.fixture
def native_report():
    return dict(agents=[],channel='stable',configSources={'layers':[]},
        cwd='/AUTHORED_ONLY/workspace',externalCompat={'cells':[],'remoteSettingsLoaded':False},
        grokVersion='1.0.46',hooks=[],loginPolicy={'apiKeyAuthDisabled':False,
            'disableApiKeyAuth':None,'forceLoginTeamUuid':None},lspServers=[],marketplaces=[],mcpServers=[],
        permissions={'claudeBypassLockAdvisory':False,'loaded':0,'managedMarketplaces':[],
            'managedSettingsActive':False,'managedSettingsExists':False,
            'managedSettingsPath':'/AUTHORED_ONLY/managed-config','marketplaceAllowlist':[],
            'marketplaceLockdownSources':[],'mcpLockdownSources':[],'mcpManagedServersOnly':'none',
            'mcpServerAllowlist':[],'skipped':[],'sources':[]},plugins=[],projectInstructions=[],
        projectRoot=None,projectTrusted=False,skills=[])


@pytest.fixture
def harness(tmp_path,native_report):
    from ipfs_accelerate_py.cli_runtime import grok_pruning as core
    root=tmp_path/'isolated';home=root/'worker-home';grok_home=home/'.grok'
    grok_home.mkdir(parents=True)
    binary=root/'authored-grok';marker=root/'inspect-started'
    env={'HOME':str(home),'GROK_HOME':str(grok_home),'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'}
    dispatches=[]
    def install(*,payload=None,body=None,exit_code=0):
        value=native_report if payload is None else payload
        program=f'#!{sys.executable}\nimport json,os,sys,time\nfrom pathlib import Path\n'
        program+="assert sys.argv[1:]==['inspect','--json']\n"
        program+=f'Path({str(marker)!r}).write_text(str(os.getpid()))\n'
        program+=body if body is not None else f'print({json.dumps(value)!r},flush=True)\n'
        program+=f'raise SystemExit({exit_code})\n'
        binary.write_text(program);binary.chmod(0o700)
    def dispatch(*,timeout=2):
        stdout,stderr,returncode=core._run_inspect([str(binary)],env,timeout)
        receipt=core.observe_grok_pruning_inspection('grok-pruning-off@1',stdout=stdout,stderr=stderr,returncode=returncode)
        dispatches.append('AUTHORED_ONLY_PROVIDER_SENTINEL')
        return receipt
    return dict(core=core,root=root,binary=binary,marker=marker,install=install,
        dispatch=dispatch,dispatches=dispatches,env=env)


def test_successful_inspection_is_config_only_and_exports_no_native_text(harness):
    harness['install']()
    result=harness['dispatch']()
    assert len(harness['dispatches'])==1 and harness['marker'].is_file()
    serialized=json.dumps(result,sort_keys=True)
    assert 'AUTHORED_ONLY' not in serialized
    assert str(harness['root']) not in serialized
    assert result['effective_settings_verified'] is False
    assert result['token_savings_established'] is False
    assert result['execution_authority'] is False
    assert result['schema']=='native-grok-pruning-effective-observation@1'
    assert result['status']=='unavailable' and result['config_source_status']=='unreported'
    assert result['inspect_schema_recognized'] is True and result['native_exit_code']==0


@pytest.mark.parametrize('exit_code',[1,7])
def test_failed_inspect_never_reaches_provider_sentinel(harness,exit_code):
    harness['install'](exit_code=exit_code)
    with pytest.raises((ValueError,RuntimeError,OSError)):
        harness['dispatch']()
    assert harness['marker'].is_file() and not harness['dispatches']


@pytest.mark.parametrize('channel',['stdout','stderr'])
def test_native_output_cap_stops_before_provider_sentinel(harness,channel):
    harness['install'](body=f"sys.{channel}.buffer.write(b'AUTHORED_ONLY_'*100000)\nsys.{channel}.flush()\ntime.sleep(30)\n")
    with pytest.raises((ValueError,RuntimeError,OSError)):
        harness['dispatch']()
    assert harness['marker'].is_file() and not harness['dispatches']
    pid=int(harness['marker'].read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)


def test_timeout_reaps_authored_child_before_provider_sentinel(harness):
    harness['install'](body='time.sleep(30)\n')
    with pytest.raises((ValueError,RuntimeError,OSError)):
        harness['dispatch'](timeout=.25)
    assert harness['marker'].is_file() and not harness['dispatches']
    pid=int(harness['marker'].read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)


def test_parent_exit_with_descendant_holding_pipes_does_not_leave_running_child(harness):
    child_marker=harness['root']/'descendant-pid'
    child='import time;time.sleep(30)'
    body='import subprocess\n'
    body+=f'p=subprocess.Popen([sys.executable,"-c",{child!r}])\n'
    body+=f'Path({str(child_marker)!r}).write_text(str(p.pid))\n'
    harness['install'](body=body)
    try:
        with pytest.raises((ValueError,RuntimeError,OSError)):
            harness['dispatch'](timeout=.4)
        assert not harness['dispatches']
        pid=int(child_marker.read_text())
        for _ in range(50):
            try:
                state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
            except FileNotFoundError:
                break
            if state=='Z':break
            time.sleep(.02)
        else:
            pytest.fail('authored inspect descendant still running after bounded capture failed')
    finally:
        if child_marker.exists():
            try:os.kill(int(child_marker.read_text()),signal.SIGKILL)
            except ProcessLookupError:pass


def test_invalid_json_does_not_become_config_only_success(harness):
    harness['install'](body="print('AUTHORED_ONLY_INVALID_JSON',flush=True)\n")
    with pytest.raises((ValueError,RuntimeError,OSError)):
        harness['dispatch']()
    assert not harness['dispatches']


def test_unqualified_native_pruning_fields_fail_closed(harness,native_report):
    native_report['compaction']={'pruning':{'enabled':False,'keep_last_n_turns':3,
        'soft_trim_threshold':4000,'soft_trim_head':1500,'soft_trim_tail':1500,'hard_clear_age_turns':10}}
    harness['install'](payload=native_report)
    with pytest.raises((ValueError,RuntimeError,OSError)):
        harness['dispatch']()
    assert not harness['dispatches']

