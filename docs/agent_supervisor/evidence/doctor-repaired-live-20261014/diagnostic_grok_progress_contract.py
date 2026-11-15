"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import math
from diagnostic_grok_usage_contract import validate_grok_usage_events
LEGACY_SCHEMA = 'native-grok-stream-progress@1'
SCHEMA = 'native-grok-stream-progress@2'
MAX_FRAME_BYTES = 256 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
CALLBACK_INTERVAL_SECONDS = 5.0
_MAX_COUNT = (1 << 63) - 1
_EVENTS = ('text', 'thought', 'tool_call', 'tool_call_update', 'usage', 'plan', 'available_commands', 'end', 'error', 'max_turns_reached', 'auto_compact', 'unknown')
_TOOLS = ('read_file', 'search_replace', 'grep', 'list_dir', 'todo_write', 'run_terminal_cmd', 'unknown')
_STATUSES = ('pending', 'in_progress', 'completed', 'failed', 'unknown')
_TERMINALS = frozenset({'end', 'error', 'max_turns_reached'})
_END_FIELDS = frozenset({'type', 'stopReason', 'sessionId', 'requestId', 'usage', 'num_turns', 'modelUsage', 'usage_is_incomplete', 'total_cost_usd', 'total_cost_usd_ticks', 'cost_is_partial'})
_REASONS = frozenset({'end_turn', 'max_turns', 'max_tokens', 'refusal', 'cancelled', 'structured_output_retries', 'execution_error', 'process_error', 'schema_rejected', 'timeout', 'other_stop', 'unknown'})
_COUNTERS = ('stdout_bytes', 'stderr_bytes', 'frames', 'malformed_frames', 'oversized_frames', 'non_object_frames', 'unknown_events', 'events_after_final', 'text_bytes', 'discarded_text_bytes', 'retained_text_bytes', 'callback_errors')
_TIMES = ('first_stdout_seconds', 'last_stdout_seconds', 'first_stderr_seconds', 'last_stderr_seconds', 'first_event_seconds', 'last_event_seconds', 'final_event_seconds', 'process_exit_seconds', 'duration_seconds')
_BOOLS = ('stdout_eof', 'stderr_eof', 'process_exit_observed', 'timed_out', 'final_envelope_observed', 'text_truncated', 'stream_tainted', 'stream_complete', 'response_usable')
_CONSTANTS = {'schema': SCHEMA, 'transport': 'streaming-json', 'text_scope': 'last_usage_delimited_text_segment', 'observation_only': True, 'completion_authority': False, 'execution_authority': False, 'proof_authority': False, 'raw_provider_data_exported': False, 'event_origin_authenticated': False}
_KEYS = frozenset((*_CONSTANTS, *_COUNTERS, *_TIMES, *_BOOLS, 'returncode', 'timeout_phase', 'final_event_type', 'final_reason_code', 'event_counts', 'tool_name_counts', 'tool_status_counts'))

def _number(value):
    return type(value) in (int, float) and 0 <= value <= _MAX_COUNT and math.isfinite(value)

def validate_grok_stream_progress(value: object) -> dict:
    """Copy one closed diagnostic snapshot; reject extra fields and authority."""
    if type(value) is not dict:
        raise ValueError('invalid Grok progress fields')
    schema = value.get('schema')
    if type(schema) is not str or schema not in {LEGACY_SCHEMA, SCHEMA}:
        raise ValueError('invalid Grok progress schema')
    keys = _KEYS if schema == LEGACY_SCHEMA else _KEYS | {'usage_observations'}
    if set(value) != keys:
        raise ValueError('invalid Grok progress fields')
    for key, expected in _CONSTANTS.items():
        if key == 'schema':
            expected = schema
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError('invalid Grok progress constants')
    for key in _COUNTERS:
        if type(value[key]) is not int or not 0 <= value[key] <= _MAX_COUNT:
            raise ValueError('invalid Grok progress counter')
    for key in _TIMES:
        if value[key] is not None and (not _number(value[key])):
            raise ValueError('invalid Grok progress time')
    for key in _BOOLS:
        if type(value[key]) is not bool:
            raise ValueError('invalid Grok progress flag')
    for key, names in (('event_counts', _EVENTS), ('tool_name_counts', _TOOLS), ('tool_status_counts', _STATUSES)):
        rows = value[key]
        if type(rows) is not dict or set(rows) != set(names) or any((type(n) is not int or not 0 <= n <= _MAX_COUNT for n in rows.values())):
            raise ValueError('invalid Grok progress buckets')
    code = value['returncode']
    if code is not None and (type(code) is not int or not -(1 << 31) <= code < 1 << 31):
        raise ValueError('invalid Grok progress exit code')
    if type(value['timeout_phase']) is not str or value['timeout_phase'] not in {'none', 'process_wait', 'pipe_drain'}:
        raise ValueError('invalid Grok progress timeout phase')
    terminal = value['final_event_type']
    if terminal is not None and (type(terminal) is not str or terminal not in _TERMINALS):
        raise ValueError('invalid Grok progress final type')
    if type(value['final_reason_code']) is not str or value['final_reason_code'] not in _REASONS:
        raise ValueError('invalid Grok progress final reason')
    if value['timed_out'] != (value['timeout_phase'] != 'none'):
        raise ValueError('inconsistent Grok progress timeout')
    if value['process_exit_observed'] != (code is not None) or value['process_exit_observed'] != (value['process_exit_seconds'] is not None):
        raise ValueError('inconsistent Grok progress process exit')
    if value['final_envelope_observed'] != (terminal is not None) or value['final_envelope_observed'] != (value['final_event_seconds'] is not None):
        raise ValueError('inconsistent Grok progress final event')
    complete = value['final_envelope_observed'] and (not value['stream_tainted']) and (not value['text_truncated']) and value['stdout_eof'] and value['stderr_eof'] and value['process_exit_observed'] and (not value['timed_out'])
    if value['stream_complete'] != complete or value['response_usable'] != (complete and code == 0 and (terminal == 'end') and (value['final_reason_code'] == 'end_turn') and (value['retained_text_bytes'] > 0)):
        raise ValueError('inconsistent Grok progress completion')
    result = {key: dict(item) if type(item) is dict else item for key, item in value.items()}
    if schema == SCHEMA:
        result['usage_observations'] = validate_grok_usage_events(value['usage_observations'], observed_events=value['event_counts']['usage'], frames=value['frames'], duration_seconds=value['duration_seconds'], final_event_seconds=value['final_event_seconds'], events_after_final=value['events_after_final'])
    return result

def grok_stream_progress_summary(value: object) -> dict:
    """Project count-only callback telemetry, retaining event rows only at exit.

    Repeating the entire bounded history every five seconds can exceed the
    outer runtime's log retention bound. The historical @1 contract precisely
    describes this projection; it does not invent missing per-event values.
    """
    result = validate_grok_stream_progress(value)
    result.pop('usage_observations', None)
    result['schema'] = LEGACY_SCHEMA
    return validate_grok_stream_progress(result)
