"""Pure authored progress fixtures exercise the export boundary, not providers."""
import copy
import json
import unittest

import diagnostic_grok_progress_contract as progress_contract
import diagnostic_grok_usage_contract as usage_contract
import read_pilot
from safe_diagnostics import grok_progress


def rejected_progress():
    state={**progress_contract._CONSTANTS,**dict.fromkeys(progress_contract._COUNTERS,0),
        **dict.fromkeys(progress_contract._TIMES),**dict.fromkeys(progress_contract._BOOLS,False),
        'returncode':1,'timeout_phase':'none','final_event_type':'max_turns_reached','final_reason_code':'max_turns',
        'event_counts':dict.fromkeys(progress_contract._EVENTS,0),'tool_name_counts':dict.fromkeys(progress_contract._TOOLS,0),
        'tool_status_counts':dict.fromkeys(progress_contract._STATUSES,0),
        'usage_observations':{**usage_contract._CONSTANTS,'observed_events':0,'retained_events':0,'omitted_events':0,'rows':[]}}
    state.update(frames=2,events_after_final=1,final_event_seconds=1,last_event_seconds=2,duration_seconds=3,
        process_exit_seconds=3,final_envelope_observed=True,process_exit_observed=True,stdout_eof=True,stderr_eof=True,stream_tainted=True)
    state['event_counts'].update(max_turns_reached=1,end=1)
    shapes=dict.fromkeys(progress_contract._TERMINAL_SHAPE_FIELDS,'missing');shapes.update(type='string',stopReason='string',sessionId='null')
    prior=dict.fromkeys(progress_contract._TERMINAL_SHAPE_FIELDS,'missing');prior.update(type='string')
    state['terminal_rejection']=dict(schema='native-grok-terminal-rejection@1',reason='invalid_end_field_shape',
        frame_ordinal=2,elapsed_seconds=2,event_type='end',prior_terminal_type='max_turns_reached',prior_reason_code='max_turns',
        incoming_reason_code='max_turns',incoming_stop_reason='max_turn_requests',field_shapes=shapes,prior_field_shapes=prior,
        unexpected_fields=[],unknown_field_count=0,rejected_fields=['sessionId'],observation_only=True,
        raw_provider_data_exported=False,completion_authority=False)
    return state


class TerminalRejectionReaderTests(unittest.TestCase):
    def test_schema3_retains_validated_diagnostic_separately_from_count_summary(self):
        source=rejected_progress();result=grok_progress(source)
        self.assertEqual(result['status'],'validated_retained_observation')
        self.assertEqual(result['terminal_rejection'],source['terminal_rejection'])
        self.assertEqual(result['progress']['schema'],'native-grok-stream-progress@1')
        self.assertNotIn('terminal_rejection',result['progress'])
        self.assertFalse(result['usage_event_values_exported'])
        result['terminal_rejection']['rejected_fields'].clear()
        result['terminal_rejection']['field_shapes']['sessionId']='string'
        self.assertEqual(source['terminal_rejection']['rejected_fields'],['sessionId'])
        self.assertEqual(source['terminal_rejection']['field_shapes']['sessionId'],'null')

    def test_historical_receipts_do_not_invent_rejection_observations(self):
        for schema in ('native-grok-stream-progress@1','native-grok-stream-progress@2'):
            source=rejected_progress();source['schema']=schema;source.pop('terminal_rejection')
            if schema.endswith('@1'):source.pop('usage_observations')
            result=grok_progress(source)
            self.assertEqual(result['status'],'validated_retained_observation')
            self.assertNotIn('terminal_rejection',result)

    def test_invalid_or_open_nested_diagnostics_do_not_export_raw_names_or_values(self):
        for mutation in (
            lambda s:s['terminal_rejection'].update(raw_end='PRIVATE_PROMPT'),
            lambda s:s['terminal_rejection']['field_shapes'].update(PRIVATE_KEY='string'),
            lambda s:s['terminal_rejection']['field_shapes'].update(sessionId='PRIVATE_VALUE'),
            lambda s:s['terminal_rejection'].update(incoming_stop_reason='PRIVATE_REASON'),
            lambda s:s['terminal_rejection'].update(unexpected_fields=['PRIVATE_KEY']),
            lambda s:s['terminal_rejection'].update(completion_authority=True),
            lambda s:s['terminal_rejection'].update(frame_ordinal=1),
        ):
            source=rejected_progress();mutation(source);result=grok_progress(source)
            self.assertEqual(result,{'status':'invalid'})
            self.assertNotIn('PRIVATE',json.dumps(result))

    def test_summary_keeps_usage_values_separate_and_never_sums_events(self):
        source=rejected_progress();source['frames']=3;source['terminal_rejection']['frame_ordinal']=3
        source['event_counts']['usage']=1
        event=dict(event_ordinal=1,frame_ordinal=1,elapsed_seconds=.5,after_final=False,
            stop_reason='tool_use',counter_status='valid',usage_container='mapping',invalid_counter_fields=0,
            usage={key:(123456 if key=='input_tokens' else None) for key in usage_contract.USAGE_FIELDS})
        source['usage_observations'].update(observed_events=1,retained_events=1,rows=[event])
        result=grok_progress(source)
        self.assertEqual(result['status'],'validated_retained_observation')
        self.assertNotIn('123456',json.dumps(result))
        self.assertNotIn('rows',result['progress']['usage_event_observation'])
        self.assertFalse(result['progress']['usage_event_observation']['summation_permitted'])

    def test_full_report_preserves_rejection_without_token_or_completion_claim(self):
        report=dict(provider_invocations=[dict(invocation_id='a'*32,phase='coding',provider='grok_cli',status='failed',
            router_calls=1,native_grok_progress=rejected_progress())])
        result=read_pilot.summarize_report(report)
        self.assertEqual(result['invocations'][0]['native_grok_progress']['terminal_rejection']['reason'],'invalid_end_field_shape')
        self.assertIsNone(result['all_phase_usage']['observed_subtotals']['total_tokens'])
        self.assertIsNone(result['all_phase_usage']['complete_native_totals']['total_tokens'])
        self.assertFalse(result['task_completed_reported'])

if __name__=='__main__':unittest.main()
