"""Authored closed public-validation reports only; no runtime imports or I/O."""
import copy
import hashlib
import json
import unittest

import read_pilot as reader
import safe_public_validation as safe
from diagnostic_public_validation_contract import FLAGS, OBSERVATION_SCHEMA
from diagnostic_public_check_contract import PUBLIC_ACCEPTANCE_RESULT_SCHEMA, PUBLIC_VALIDATION_PROFILE, PUBLIC_CHECKS, PUBLIC_VALIDATION_KEYS
from diagnostic_doctor_dispatch_contract import _OBSERVATION_FLAGS, _DIGEST_FIELDS
from test_read_pilot import two_round_report, complete_aggregate


def validation(number, status):
    names=('.supervisor-instruction.md','.supervisor-task-profile.json','.supervisor-public-smoke.py','eval.py','eigen.py')
    rows=[]
    for check,key in zip(PUBLIC_CHECKS,PUBLIC_VALIDATION_KEYS):
        failed=check=='performance' and status=='failed'
        reason='candidate_behavior_invalid' if failed else 'faster_than_reference' if check=='performance' else 'obligations_satisfied'
        measured=[] if reason!='faster_than_reference' else [dict(size=size,batch=batch,candidate_median_seconds=1e-6,
            reference_median_seconds=2e-6) for size in (2,4,6,8,10) for batch in range(3)]
        result=dict(schema=PUBLIC_ACCEPTANCE_RESULT_SCHEMA,profile=PUBLIC_VALIDATION_PROFILE,check=check,
            status='failed' if failed else 'passed',reason=reason,measurements=measured)
        rows.append(dict(validation_key=key,returncode=1 if failed else 0,result=result,
            status=result['status'],reason=reason))
    return dict(schema=OBSERVATION_SCHEMA,contract_sha256='1'*64,original_deadline_ns=10**12,
        actual_returncode=0,round_number=number,session_id=('a' if number==1 else 'd')*32,
        descriptor_sha256=('b' if number==1 else 'e')*64,workspace_generation=str(number)*64,
        source_hashes={'eigen.py':'2'*64},workspace_states={name:dict(sha256='2'*64,bytes=8,mode=0o640,uid=1000,gid=1001) for name in names},
        status=status,failed_obligations=[PUBLIC_VALIDATION_KEYS[2]] if status=='failed' else [],checks=rows,**FLAGS)


def report():
    value=two_round_report();value['doctor_coding_policy']=safe.PUBLIC_VALIDATION_POLICY
    observations=[validation(1,'failed'),validation(2,'passed')]
    accounting=value['candidate_attempt_accounting']
    accounting.update(doctor_coding_policy=safe.PUBLIC_VALIDATION_POLICY,owner_validation_outcome='passed',owner_returncode=0,
        public_validation_observations=copy.deepcopy(observations))
    marker=dict(schema='owner-candidate-rounds-observation@2',rounds=[dict(round_number=n,process_started=True,
        session_id=observations[n-1]['session_id'],descriptor_sha256=observations[n-1]['descriptor_sha256'],returncode=0,
        wait_completed=True,retirement_observed=True) for n in (1,2)],max_rounds=2,elapsed_seconds=10,
        outcome='workers_returned',failure_type=None,completion_authority=False,proof_authority=False,
        source_obligations_discharged=0,accepted_index_published=False,public_validation_observations=observations,
        owner_validation_outcome='passed',owner_returncode=0,
        owner_lifecycle_retention=dict(path='PRIVATE OWNER PATH',sha256='3'*64,bytes=1000))
    value['owner_candidate_rounds']=[marker]
    budget=dict(max_calls=1,max_tokens=100,max_rounds=1,max_cost_units=1)
    doctor=dict(schema='supervisor-doctor-candidate-dispatch-observation@2',policy_id=safe.PUBLIC_VALIDATION_POLICY,
        proposal_budget=budget,doctor_advisory_bytes=100,source_files=2,task_cid='task-cid',context_cid='context-cid',
        native_capsule_id='capsule-id',**{key:'4'*64 for key in _DIGEST_FIELDS},**_OBSERVATION_FLAGS)
    doctor.update(coding_policy_sha256='d'*64,doctor_artifact_sha256='f'*64,doctor_advisory_sha256='e'*64,
        proposal_budget_sha256=hashlib.sha256(reader.canonical(budget)).hexdigest(),
        public_validation_contract_sha256=observations[0]['contract_sha256'],
        public_validation_observation_sha256=hashlib.sha256(reader.canonical(observations[0])).hexdigest(),
        actual_prior_returncode=0,public_validation_status='failed')
    value['provider_invocations'][2]['candidate_round_dispatch']['doctor_candidate_dispatch']=doctor
    return value


class PublicValidationReaderTests(unittest.TestCase):
    def test_closed_public_checks_join_real_returncodes_without_private_data(self):
        value=report();result=reader.summarize_report(value,complete_aggregate())
        public=result['public_validation']
        self.assertEqual(public['status'],'validated_retained_observation')
        self.assertEqual([r['actual_returncode'] for r in public['rounds']],[0,0])
        self.assertEqual([r['status'] for r in public['rounds']],['failed','passed'])
        self.assertEqual(public['owner_returncode'],0)
        self.assertEqual(len(public['rounds'][1]['checks'][2]['measurements']),15)
        self.assertTrue(result['continuation']['continuation_supported_by_retained_report'])
        self.assertFalse(public['completion_authority'])
        self.assertFalse(public['independent_owner_replay_performed'])
        encoded=json.dumps(result)
        for private in ('PRIVATE OWNER PATH','workspace_states','source_hashes','eigen.py'):
            self.assertNotIn(private,encoded)
        public['rounds'][0]['checks'][0]['reason']='changed'
        self.assertEqual(value['owner_candidate_rounds'][0]['public_validation_observations'][0]['checks'][0]['reason'],'obligations_satisfied')

    def test_legacy_has_no_invented_public_checks(self):
        result=reader.summarize_report(two_round_report(),complete_aggregate())
        self.assertNotIn('public_validation',result)
        self.assertTrue(result['continuation']['continuation_supported_by_retained_report'])

    def test_missing_or_incomplete_validation_never_grants_continuation(self):
        for change in ('no_marker','duplicate','wrong_schema','missing_second_receipt','new_policy_without_observations'):
            with self.subTest(change=change):
                value=report()
                if change=='no_marker':value.pop('owner_candidate_rounds')
                elif change=='duplicate':value['owner_candidate_rounds']*=2
                elif change=='wrong_schema':value['owner_candidate_rounds'][0]['schema']='owner-candidate-rounds-observation@1'
                elif change=='missing_second_receipt':value['provider_invocations'].pop()
                else:value['owner_candidate_rounds'][0]['public_validation_observations']=[]
                self.assertFalse(reader.summarize_report(value)['continuation']['continuation_supported_by_retained_report'])

    def test_closed_shape_and_types_cannot_export_authored_data(self):
        for change in ('extra','reason','key','bool_code','bool_round','owner_code','authority','source_field','unknown_phase'):
            with self.subTest(change=change):
                value=report();marker=value['owner_candidate_rounds'][0];observation=marker['public_validation_observations'][0]
                if change=='extra':observation['prompt']='PRIVATE authored prompt'
                elif change=='reason':observation['checks'][0]['reason']='PRIVATE authored prompt'
                elif change=='key':observation['checks'][0]['validation_key']='PRIVATE authored prompt'
                elif change=='bool_code':observation['actual_returncode']=False
                elif change=='bool_round':observation['round_number']=True
                elif change=='owner_code':marker['owner_returncode']=False
                elif change=='authority':observation['completion_authority']=True
                elif change=='source_field':observation['workspace_states']['eigen.py']['secret']='PRIVATE authored prompt'
                else:marker['PUBLIC raw phase']=1
                result=reader.summarize_report(value)
                self.assertEqual(result['public_validation']['status'],'invalid_or_unavailable')
                self.assertFalse(result['continuation']['continuation_supported_by_retained_report'])
                self.assertNotIn('PRIVATE',json.dumps(result))
                self.assertNotIn('PUBLIC raw phase',json.dumps(result))

    def test_dispatch_receipt_must_join_exact_first_public_observation(self):
        for change in ('contract','observation','returncode','status','budget','schema','authority'):
            with self.subTest(change=change):
                value=report();doctor=value['provider_invocations'][2]['candidate_round_dispatch']['doctor_candidate_dispatch']
                if change=='contract':doctor['public_validation_contract_sha256']='9'*64
                elif change=='observation':doctor['public_validation_observation_sha256']='9'*64
                elif change=='returncode':doctor['actual_prior_returncode']=1
                elif change=='status':doctor['public_validation_status']='inconclusive'
                elif change=='budget':doctor['proposal_budget']['max_calls']=True
                elif change=='schema':doctor['schema']='supervisor-doctor-candidate-dispatch-observation@1'
                else:doctor['completion_authority']=True
                self.assertFalse(reader.summarize_report(value)['continuation']['continuation_supported_by_retained_report'])

    def test_validation_status_never_fills_missing_native_usage(self):
        value=report()
        for row in value['provider_invocations']:
            if row['phase']=='coding':row.pop('native_rollout_usage')
        result=reader.summarize_report(value)
        self.assertEqual(result['public_validation']['owner_validation_outcome'],'passed')
        self.assertEqual(result['all_phase_usage']['observed_subtotals']['total_tokens'],37)
        self.assertIsNone(result['usage_by_phase']['coding']['observed_subtotals']['total_tokens'])
        self.assertIsNone(result['all_phase_usage']['complete_native_totals']['total_tokens'])


if __name__=='__main__':unittest.main()
