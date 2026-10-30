"""Real authored Python children exercise terminal-only cleanup; no providers."""
import json
import os
import sys
import time

import pytest

from ipfs_accelerate_py.cli_runtime import grok_stream_capture as stream


def frames(*events):
    return b''.join(json.dumps(event).encode() + b'\n' for event in events)


def end(**values):
    return {'type': 'end', 'stopReason': 'end_turn', **values}


def authored(tmp_path, before, *, terminal=None, after='', timeout=.4):
    script = tmp_path / 'authored.py'
    script.write_text('import os,time\nfrom pathlib import Path\n'
                      "Path('pid').write_text(str(os.getpid()))\n" + before + '\n'
                      + (f'os.write(1, {terminal!r})\n' if terminal is not None else '') + after)
    result = stream.capture_grok_stream([sys.executable, str(script)], env=os.environ.copy(),
                                        cwd=tmp_path, timeout=timeout)
    assert stream.validate_grok_stream_progress(result.progress) == result.progress
    with pytest.raises(ProcessLookupError): os.kill(int((tmp_path / 'pid').read_text()), 0)
    return result


def test_clean_terminal_before_deadline_waits_for_natural_zero_exit_and_both_eof(tmp_path):
    result = authored(tmp_path, 'time.sleep(.12)',
                      terminal=frames({'type': 'text', 'data': 'complete'}, end()),
                      after='time.sleep(.4)\n', timeout=.4)
    assert result.progress['final_event_seconds'] < .4
    assert .4 < result.progress['process_exit_seconds'] < .8
    assert result.returncode == 0 and not result.timed_out
    assert result.response_text == 'complete'
    assert result.progress['stdout_eof'] is result.progress['stderr_eof'] is True
    assert result.progress['response_usable'] is True


def test_no_terminal_never_receives_cleanup_grace(tmp_path):
    result = authored(tmp_path, '', terminal=frames({'type': 'text', 'data': 'partial'}),
                      after='time.sleep(5)\n', timeout=.3)
    assert result.timed_out and not result.progress['response_usable']
    assert result.progress['duration_seconds'] < .5


def test_terminal_observed_after_deadline_cannot_sneak_past_exit_and_eof_break(tmp_path, monkeypatch):
    original_read = os.read
    delayed = False
    def slow_eof(fd, size):
        nonlocal delayed
        data = original_read(fd, size)
        if size == 64 * 1024 and not data and not delayed:
            delayed = True
            time.sleep(.3)  # Real parent pipe-handling delay, not a mocked child or clock.
        return data
    monkeypatch.setattr(stream.os, 'read', slow_eof)
    payload = frames({'type': 'text', 'data': 'complete'}, end()).rstrip(b'\n')
    result = authored(tmp_path, '', terminal=payload, after='', timeout=.2)
    assert delayed and result.progress['final_event_seconds'] >= .2
    assert result.returncode == 0 and result.timed_out
    assert result.response_text == result.stdout == ''


@pytest.mark.parametrize('terminal', [end(stopReason='max_tokens'), end(stopReason='cancelled'),
                                     {'type': 'error', 'message': 'authored error'}])
def test_non_success_terminal_never_receives_cleanup_grace(tmp_path, terminal):
    result = authored(tmp_path, '', terminal=frames({'type': 'text', 'data': 'partial'}, terminal),
                      after='time.sleep(5)\n', timeout=.3)
    assert result.timed_out and not result.progress['response_usable']
    assert result.progress['duration_seconds'] < .5


def test_tainted_response_never_receives_cleanup_grace(tmp_path):
    result = authored(tmp_path, '', terminal=b'bad-json\n' + frames({'type': 'text', 'data': 'partial'}, end()),
                      after='time.sleep(5)\n', timeout=.3)
    assert result.timed_out and result.progress['stream_tainted']
    assert result.progress['duration_seconds'] < .5


def test_terminal_without_nonempty_response_never_receives_cleanup_grace(tmp_path):
    result = authored(tmp_path, '', terminal=frames(end()), after='time.sleep(5)\n', timeout=.3)
    assert result.timed_out and result.progress['duration_seconds'] < .5
    assert not result.progress['response_usable']


def test_unfinished_frame_after_terminal_cannot_qualify_for_grace(tmp_path):
    result = authored(tmp_path, '', terminal=frames({'type': 'text', 'data': 'done'}, end()) + b'garbage',
                      after='time.sleep(5)\n', timeout=.3)
    assert result.timed_out and result.progress['duration_seconds'] < .5
    assert not result.progress['response_usable']


def test_late_protocol_event_during_cleanup_invalidates_earlier_terminal(tmp_path):
    late = frames({'type': 'text', 'data': 'late forbidden output'})
    result = authored(tmp_path, 'time.sleep(.1)', terminal=frames({'type': 'text', 'data': 'done'}, end()),
                      after=f'time.sleep(.4)\nos.write(1, {late!r})\n', timeout=.4)
    assert result.progress['events_after_final'] == 1
    assert result.progress['stream_tainted'] and not result.progress['response_usable']
    assert result.stdout == result.response_text == ''


def test_natural_nonzero_exit_after_terminal_never_yields_success(tmp_path):
    result = authored(tmp_path, 'time.sleep(.1)', terminal=frames({'type': 'text', 'data': 'done'}, end()),
                      after='time.sleep(.4)\nos._exit(7)\n', timeout=.4)
    assert result.returncode == 7 and not result.progress['response_usable']
    assert result.response_text == ''


def test_terminal_cannot_keep_never_exiting_direct_child_alive_past_fixed_cleanup(tmp_path):
    result = authored(tmp_path, '', terminal=frames({'type': 'text', 'data': 'done'}, end()),
                      after='time.sleep(5)\n', timeout=.3)
    assert result.timed_out and result.returncode != 0
    assert .5 < result.progress['duration_seconds'] < .9
    assert result.progress['timeout_phase'] == 'process_wait'


def test_exact_terminal_boundary_is_not_before_deadline():
    parser = stream.GrokStreamParser()
    parser.feed(frames({'type': 'text', 'data': 'done'}, end()), elapsed_seconds=.3)
    assert not stream._terminal_drain_allowed(parser, .3)
    assert stream._terminal_drain_allowed(parser, .300001)


def test_fixed_cleanup_limit_is_not_caller_configurable():
    import inspect
    assert stream.TERMINAL_DRAIN_SECONDS == 3
    assert 'terminal_drain' not in inspect.signature(stream.capture_grok_stream).parameters
