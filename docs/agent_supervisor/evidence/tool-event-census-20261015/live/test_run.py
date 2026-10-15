"""Authored launcher controls. No Docker, providers, credentials or task bodies."""
import importlib.util
import json
from pathlib import Path
import types

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = Path(__file__).with_name('run.py')
    spec = importlib.util.spec_from_file_location('authored_working_census_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'A', tmp_path)
    return module


def test_missing_freeze_refuses_before_any_command(launcher, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('command dispatched'))
    with pytest.raises(FileNotFoundError):
        launcher.step('execute')
    assert list(launcher.A.iterdir()) == []


def test_insufficient_disk_refuses_before_start_or_command(launcher, monkeypatch):
    monkeypatch.setattr(launcher, 'check', lambda: {'minimum_free_bytes': 32768})
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda p: types.SimpleNamespace(free=32767))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('command dispatched'))
    with pytest.raises(AssertionError):
        launcher.step('execute')
    assert list(launcher.A.iterdir()) == []


def test_post_step_pin_failure_is_not_success(launcher, monkeypatch):
    calls = []

    def check():
        calls.append(True)
        if len(calls) > 1:
            raise ValueError('authored pin drift')
        return {'minimum_free_bytes': 1}

    monkeypatch.setattr(launcher, 'check', check)
    monkeypatch.setattr(launcher, 'sha', lambda p: 'a' * 64)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda p: types.SimpleNamespace(free=32768))
    monkeypatch.setattr(launcher, 'verify', lambda r: {'all_archive_members_verified': True})
    assert launcher.step('verify') == 1
    report = json.loads((launcher.A / 'verify-result.json').read_text())
    assert report['exit_code'] == 1 and report['error_type'] == 'ValueError'
    assert 'frozen_inputs_unchanged' not in report


def test_repeated_operation_refuses_before_body(launcher, monkeypatch):
    monkeypatch.setattr(launcher, 'check', lambda: {'minimum_free_bytes': 1})
    monkeypatch.setattr(launcher, 'sha', lambda p: 'a' * 64)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda p: types.SimpleNamespace(free=32768))
    monkeypatch.setattr(launcher, 'verify', lambda r: pytest.fail('operation repeated'))
    (launcher.A / 'verify-start.json').write_text('{}')
    with pytest.raises(FileExistsError):
        launcher.step('verify')


def test_provider_keys_scrubbed_without_reading_credentials(launcher, monkeypatch):
    manifest = {'scrub_inherited_names': ['XAI_API_KEY', 'HF_TOKEN'],
                'environment_overrides': {'PYTHONDONTWRITEBYTECODE': '1'}}
    (launcher.A / 'environment.json').write_text(json.dumps(manifest))
    monkeypatch.setenv('XAI_API_KEY', 'authored-never-a-real-key')
    monkeypatch.setenv('HF_TOKEN', 'authored-never-a-real-key')
    env = launcher.environment()
    assert 'XAI_API_KEY' not in env and 'HF_TOKEN' not in env
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'


def test_diagnostic_controls_match_single_invocation_default_recipe(launcher):
    recipe=json.loads(Path(__file__).with_name('recipe-draft.json').read_text())
    launcher.validate_diagnostic_controls(recipe)


@pytest.mark.parametrize('field,value',[('candidate_rounds',2),('grok_coding_max_turns',9),
    ('provider_fallback',True),('automatic_retries',1),('official_verifier_enabled',True),
    ('max_coding_invocations',True),('working_set_census_only',True),
    ('semantic_working_set','capsule-references@1'),('disable_intent_autoencoder',False)])
def test_diagnostic_control_drift_refuses(launcher,field,value):
    recipe=json.loads(Path(__file__).with_name('recipe-draft.json').read_text());recipe[field]=value
    with pytest.raises(AssertionError):launcher.validate_diagnostic_controls(recipe)
