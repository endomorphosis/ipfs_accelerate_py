"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import math
from diagnostic_grok_usage_contract import validate_grok_usage_events
LEGACY_SCHEMA = 'native-grok-stream-progress@1'
USAGE_SCHEMA = 'native-grok-stream-progress@2'
SCHEMA = 'native-grok-stream-progress@3'
MAX_FRAME_BYTES = 256 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
CALLBACK_INTERVAL_SECONDS = 5.0
_MAX_COUNT = (1 << 63) - 1
_EVENTS = ('text', 'thought', 'tool_call', 'tool_call_update', 'usage', 'plan', 'available_commands', 'end', 'error', 'max_turns_reached', 'auto_compact', 'unknown')
_TOOLS = ('read_file', 'search_replace', 'grep', 'list_dir', 'todo_write', 'run_terminal_cmd', 'unknown')
_STATUSES = ('pending', 'in_progress', 'completed', 'failed', 'unknown')
_TERMINALS = frozenset({'end', 'error', 'max_turns_reached'})
_END_FIELDS = frozenset({'type', 'stopReason', 'sessionId', 'requestId', 'usage', 'num_turns', 'modelUsage', 'usage_is_incomplete', 'total_cost_usd', 'total_cost_usd_ticks', 'cost_is_partial'})
_TERMINAL_SHAPE_FIELDS = _END_FIELDS | {'stop_reason', 'subtype', 'is_error', 'message', 'text', 'thought', 'cancellationCategory'}
_JSON_SHAPES = frozenset({'missing', 'null', 'boolean', 'integer', 'number', 'string', 'object', 'array', 'other'})
_TERMINAL_REJECTIONS = frozenset({'not_end_event', 'duplicate_end_trailer', 'prior_payload_unavailable', 'unknown_prior_reason', 'second_final_envelope', 'unknown_end_fields', 'end_reason_mismatch', 'invalid_end_field_shape', 'missing_frozen_fields', 'conflicting_frozen_fields'})
_TERMINAL_STOP_REASONS = frozenset({'end_turn', 'stop', 'max_tokens', 'max_turn_requests', 'refusal', 'cancelled', 'tool_use', 'pause_turn', 'stop_sequence', 'EndTurn', 'MaxTokens', 'MaxTurnRequests', 'Refusal', 'Cancelled', 'unknown'})
_REASONS = frozenset({'end_turn', 'max_turns', 'max_tokens', 'refusal', 'cancelled', 'structured_output_retries', 'execution_error', 'process_error', 'schema_rejected', 'timeout', 'other_stop', 'unknown'})
_COUNTERS = ('stdout_bytes', 'stderr_bytes', 'frames', 'malformed_frames', 'oversized_frames', 'non_object_frames', 'unknown_events', 'events_after_final', 'text_bytes', 'discarded_text_bytes', 'retained_text_bytes', 'callback_errors')
_TIMES = ('first_stdout_seconds', 'last_stdout_seconds', 'first_stderr_seconds', 'last_stderr_seconds', 'first_event_seconds', 'last_event_seconds', 'final_event_seconds', 'process_exit_seconds', 'duration_seconds')
_BOOLS = ('stdout_eof', 'stderr_eof', 'process_exit_observed', 'timed_out', 'final_envelope_observed', 'text_truncated', 'stream_tainted', 'stream_complete', 'response_usable')
_CONSTANTS = {'schema': SCHEMA, 'transport': 'streaming-json', 'text_scope': 'last_usage_delimited_text_segment', 'observation_only': True, 'completion_authority': False, 'execution_authority': False, 'proof_authority': False, 'raw_provider_data_exported': False, 'event_origin_authenticated': False}
_KEYS = frozenset((*_CONSTANTS, *_COUNTERS, *_TIMES, *_BOOLS, 'returncode', 'timeout_phase', 'final_event_type', 'final_reason_code', 'event_counts', 'tool_name_counts', 'tool_status_counts'))

def _number(value):
    return type(value) in (int, float) and 0 <= value <= _MAX_COUNT and math.isfinite(value)

def _validate_terminal_rejection(value):
    """Copy one fixed vocabulary diagnostic, without provider names or values."""
    keys = {'schema', 'reason', 'frame_ordinal', 'elapsed_seconds', 'event_type', 'prior_terminal_type', 'prior_reason_code', 'incoming_reason_code', 'incoming_stop_reason', 'field_shapes', 'prior_field_shapes', 'unexpected_fields', 'unknown_field_count', 'rejected_fields', 'observation_only', 'raw_provider_data_exported', 'completion_authority'}
    if type(value) is not dict or set(value) != keys or value['schema'] != 'native-grok-terminal-rejection@1' or (value['observation_only'] is not True) or (value['raw_provider_data_exported'] is not False) or (value['completion_authority'] is not False):
        raise ValueError('invalid Grok terminal rejection fields')
    for key, allowed in (('reason', _TERMINAL_REJECTIONS), ('event_type', _EVENTS), ('prior_terminal_type', _TERMINALS), ('prior_reason_code', _REASONS), ('incoming_reason_code', _REASONS), ('incoming_stop_reason', _TERMINAL_STOP_REASONS)):
        if type(value[key]) is not str or value[key] not in allowed:
            raise ValueError('invalid Grok terminal rejection vocabulary')
    if type(value['frame_ordinal']) is not int or not 1 <= value['frame_ordinal'] <= _MAX_COUNT or type(value['unknown_field_count']) is not int or (not 0 <= value['unknown_field_count'] <= MAX_FRAME_BYTES) or (not _number(value['elapsed_seconds'])):
        raise ValueError('invalid Grok terminal rejection counters')
    for key in ('field_shapes', 'prior_field_shapes'):
        shapes = value[key]
        if type(shapes) is not dict or set(shapes) != _TERMINAL_SHAPE_FIELDS or any((type(shape) is not str or shape not in _JSON_SHAPES for shape in shapes.values())):
            raise ValueError('invalid Grok terminal rejection shapes')
    for key, allowed in (('unexpected_fields', _TERMINAL_SHAPE_FIELDS - _END_FIELDS), ('rejected_fields', _END_FIELDS)):
        fields = value[key]
        if type(fields) is not list or len(fields) > len(allowed) or any((type(field) is not str or field not in allowed for field in fields)) or (fields != sorted(set(fields))):
            raise ValueError('invalid Grok terminal rejection field list')
    return {**value, 'field_shapes': dict(value['field_shapes']), 'prior_field_shapes': dict(value['prior_field_shapes']), 'unexpected_fields': list(value['unexpected_fields']), 'rejected_fields': list(value['rejected_fields'])}

def validate_grok_stream_progress(value: object) -> dict:
    """Copy one closed diagnostic snapshot; reject extra fields and authority."""
    if type(value) is not dict:
        raise ValueError('invalid Grok progress fields')
    schema = value.get('schema')
    if type(schema) is not str or schema not in {LEGACY_SCHEMA, USAGE_SCHEMA, SCHEMA}:
        raise ValueError('invalid Grok progress schema')
    keys = _KEYS if schema == LEGACY_SCHEMA else _KEYS | {'usage_observations'}
    if schema == SCHEMA:
        keys |= {'terminal_rejection'}
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
    if schema != LEGACY_SCHEMA:
        result['usage_observations'] = validate_grok_usage_events(value['usage_observations'], observed_events=value['event_counts']['usage'], frames=value['frames'], duration_seconds=value['duration_seconds'], final_event_seconds=value['final_event_seconds'], events_after_final=value['events_after_final'])
    if schema == SCHEMA:
        rejection = value['terminal_rejection']
        if rejection is not None:
            rejection = _validate_terminal_rejection(rejection)
            if not value['stream_tainted'] or value['events_after_final'] < 1 or rejection['prior_terminal_type'] != terminal or (rejection['prior_reason_code'] != value['final_reason_code']) or (not value['frames'] - value['events_after_final'] < rejection['frame_ordinal'] <= value['frames']) or (value['event_counts'][rejection['event_type']] < 1) or (value['final_event_seconds'] is None) or (value['duration_seconds'] is None) or (not value['final_event_seconds'] <= rejection['elapsed_seconds'] <= value['duration_seconds']):
                raise ValueError('inconsistent Grok terminal rejection observation')
        result['terminal_rejection'] = rejection
    return result

def grok_stream_progress_summary(value: object) -> dict:
    """Project count-only callback telemetry, retaining event rows only at exit.

    Repeating the entire bounded history every five seconds can exceed the
    outer runtime's log retention bound. The historical @1 contract precisely
    describes this projection; it does not invent missing per-event values.
    """
    result = validate_grok_stream_progress(value)
    result.pop('usage_observations', None)
    result.pop('terminal_rejection', None)
    result['schema'] = LEGACY_SCHEMA
    return validate_grok_stream_progress(result)
