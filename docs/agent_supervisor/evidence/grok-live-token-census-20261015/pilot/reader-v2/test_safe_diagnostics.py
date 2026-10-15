"""Authored closed diagnostic fixtures; no supervisor or provider imports."""
import copy
import json
import unittest
from safe_diagnostics import FLAGS,failure_observations,grok_progress

def envelope(phase,kind):
    return dict(schema='router-implementation-error@2',error_type=kind,diagnostic=dict(
        schema='router-implementation-failure-diagnostic@1',phase=phase,
        exceptions=[dict(exception_type=kind,frames=[dict(file='router_implementation_runner.py',line=10)])],
        chain_truncated=False,completion_authority=False,automatic_retry_admitted=False,
        settlement_authority=False,provider_dispatch_observed=None))

def observed():
    child=dict(schema='router-child-failure-observations@1',status='observed_multiple_unattributed',
        diagnostics=sorted([envelope('provider_invocation','RuntimeError'),envelope('candidate_round_preflight','ValueError')],
            key=lambda row:json.dumps(row,sort_keys=True,separators=(',',':'))),observed_record_count=2,
        scope='bounded_native_log_tail',attempt_attribution_verified=False,provider_dispatch_observed=None,**FLAGS)
    diagnostic=dict(schema='database-bridge-failure-diagnostic@2',phase='terminal_failure',reason_code='portal_provider_failed',
        exceptions=[dict(exception_type='DatabasePortalBridgeError',frames=[])],chain_truncated=False,
        callback=dict(schema=None,state=None,provider_effect_state=None,native_exit=dict(present=False,returncode=None,
            reaped=None,process_group_absent=None,subreaper_children_absent=None,lifecycle_finalized=None)),
        child_reported_router_failure=child,native_provider_custody_observation=None,
        completion_authority=False,retry_authority=False,settlement_authority=False)
    def row(status,scope,diagnostic=None):return dict(status=status,scope=scope,diagnostic=diagnostic,
        matched_record_count=1 if diagnostic else 0,provider_dispatch_observed=None,**FLAGS)
    return dict(schema='terminal-native-failure-observations@1',bridge=row('observed','exact_admitted_task_and_attempt',diagnostic),
        planner_child=row('missing','planner_child_stderr'),**FLAGS)

class DiagnosticTests(unittest.TestCase):
    def test_multi_error_observation_is_not_attempt_or_dispatch_attribution(self):
        result=failure_observations(observed());self.assertEqual(result['status'],'validated_retained_observation')
        child=result['bridge']['diagnostic']['child_reported_router_failure']
        self.assertEqual(child['observed_record_count'],2);self.assertFalse(child['attempt_attribution_verified'])
        self.assertIsNone(child['provider_dispatch_observed']);self.assertFalse(child['completion_authority'])
    def test_unknown_payload_fields_and_authority_are_not_exported(self):
        for target,key,val in [('root','prompt','SECRET'),('bridge','message','SECRET'),('child','raw_log','SECRET'),('child','attempt_attribution_verified',True)]:
            value=observed();row=value if target=='root' else value['bridge']['diagnostic']
            if target=='child':row=row['child_reported_router_failure']
            row[key]=val;result=failure_observations(value)
            self.assertEqual(result['status'],'invalid');self.assertNotIn('SECRET',json.dumps(result))
    def test_contradictory_duplicate_or_noncanonical_child_rows_are_rejected(self):
        for variant in ('duplicate','order','count','exception'):
            value=observed();child=value['bridge']['diagnostic']['child_reported_router_failure']
            if variant=='duplicate':child['diagnostics'][1]=copy.deepcopy(child['diagnostics'][0])
            if variant=='order':child['diagnostics'].reverse()
            if variant=='count':child['observed_record_count']=True
            if variant=='exception':child['diagnostics'][0]['diagnostic']['exceptions'][0]['exception_type']='PRIVATE'
            self.assertEqual(failure_observations(value)['status'],'invalid')
    def test_single_legacy_child_and_missing_are_preserved(self):
        value=observed();parent=value['bridge']['diagnostic'];parent['child_reported_router_failure']=dict(
            status='observed',diagnostic=envelope('provider_invocation','RuntimeError'),observation_only=True,scope='bounded_native_log_tail')
        self.assertEqual(failure_observations(value)['status'],'validated_retained_observation')
        self.assertEqual(failure_observations(None)['status'],'unavailable')
    def test_invalid_progress_never_exports_text_or_usage_values(self):
        self.assertEqual(grok_progress({'text':'SECRET'})['status'],'invalid')
        self.assertEqual(grok_progress(None)['status'],'unavailable')
if __name__=='__main__':unittest.main()
