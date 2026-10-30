"""Grok response transport preserves literal source; no paid provider calls."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation

pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise literal source and native transport extraction")


def proposal():
    return {"edits": [{"old": '<div class="old">A &amp; B &#x41; &lt;tag&gt;</div>\n',
                       "new": '<button data-value="a &quot;b&quot;">if (x < y && y > z) \\"quoted\\"</button>\n'}]}


@pytest.mark.parametrize("literal", [
    '<!doctype html>\n<main><button title="A &amp; B">&lt;keep&gt;</button></main>',
    'if (left < right && next > last) return "<div>&amp; &#x41;</div>";',
    '- preserve leading minus\n* preserve markdown\n• preserve bullet',
    json.dumps(proposal(), ensure_ascii=False),
    '```json\n' + json.dumps(proposal(), indent=2) + '\n```',
    r'{"old":"\\u003cdiv\\u003e &amp;", "new":"line\\n\\\"quoted\\\""}',
    '*** Begin Patch\n-<div>&amp;</div>\n+<button>&lt;</button>\n*** End Patch',
])
def test_grok_cleaner_only_trims_outer_whitespace(literal):
    assert llm_router._clean_grok_cli_output(' \n' + literal + '\n\t') == literal


def test_json_hunk_values_round_trip_without_html_entity_or_escape_mutation():
    expected = proposal()
    raw = json.dumps(expected, indent=2)
    cleaned = llm_router._clean_grok_cli_output(raw)
    assert cleaned == raw
    assert json.loads(cleaned) == expected
    assert json.loads(cleaned)["edits"][0]["old"].count('<div class="old">') == 1


def test_other_provider_legacy_cleaners_keep_their_existing_display_behavior():
    raw = ' • <b>A &amp; B</b> '
    for cleaner in [llm_router._clean_codex_output, llm_router._clean_copilot_output,
                    llm_router._clean_claude_output, llm_router._clean_gemini_output,
                    llm_router._clean_mistral_vibe_output]:
        assert cleaner(raw) == 'A & B'
    assert llm_router._clean_grok_cli_output(raw) == '• <b>A &amp; B</b>'


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(llm_router, 'find_grok_cli', lambda: '/fixture/grok')
    monkeypatch.setattr(llm_router, '_cli_available', lambda _: True)
    set_last_cli_observation('grok_cli', {})
    return llm_router._get_grok_cli_provider()


@pytest.mark.parametrize('envelope', [True, False])
def test_native_nonstream_return_paths_preserve_literal_code(provider, monkeypatch, envelope):
    literal = json.dumps(proposal()) if envelope else '<section data-x="&amp;">x < y && y > z</section>'
    output = json.dumps({'text': literal, 'stopReason': 'end_turn'}) if envelope else literal
    calls = []
    def execute(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=output, stderr='')
    monkeypatch.setattr(llm_router.subprocess, 'run', execute)
    assert provider.generate('literal repair request', model_name='grok-4.7') == literal
    assert len(calls) == 1


@pytest.mark.parametrize('envelope', [True, False])
def test_custom_wrapper_return_paths_preserve_fenced_json_and_source(provider, monkeypatch, envelope):
    literal = '```json\n' + json.dumps(proposal()) + '\n```' if envelope else '<main>&amp; &#60; content</main>'
    output = json.dumps({'text': literal, 'stopReason': 'end_turn'}) if envelope else literal
    monkeypatch.setattr(llm_router, '_run_cli_command', lambda *_args, **_kwargs: output)
    assert provider.generate('literal repair request', grok_cli_cmd='wrapper --prompt {prompt}') == literal


def test_real_stream_parser_and_provider_preserve_json_hunks_with_one_end_turn(provider, tmp_path):
    literal = json.dumps(proposal(), indent=2)
    middle = literal.index('button') + 2
    frames = [{'type': 'text', 'data': literal[:middle]}, {'type': 'text', 'data': literal[middle:]},
              {'type': 'end', 'stopReason': 'end_turn', 'num_turns': 1,
               'usage': {'input_tokens': 1, 'output_tokens': 1}}]
    script = tmp_path / 'authored_native_stream.py'
    script.write_text('import json\n' + '\n'.join(f'print({json.dumps(frame)!r}, flush=True)' for frame in frames) + '\n')
    result = provider.generate('literal repair request', model_name='grok-4.7', timeout=5,
        grok_cli_cmd=[sys.executable, str(script)], grok_stream_progress=True)
    assert result == literal
    assert json.loads(result) == proposal()
    observation = get_last_cli_observation('grok_cli')
    assert observation['exit_code'] == 0 and observation['native_grok_reason_code'] == 'end_turn'
    assert observation['native_grok_progress']['response_usable'] is True
    assert observation['native_grok_progress']['event_counts']['end'] == 1
    assert observation['native_grok_progress']['tool_name_counts']['read_file'] == 0
