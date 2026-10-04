"""Pure mutation controls for resource/cost joins, with no native execution.

Controlled positive predicates are synthetic completions of retained failed02
observations. They do not qualify that run or stand in for an actual worker.
"""
from copy import deepcopy
from pathlib import Path
import json
from unittest.mock import patch
import unittest

from . import audit_codebase_signed_successor_archive as audit

FIXTURE = Path('/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/signed-successor-worker-qualification-20261003-02')
PREPARED_FIXTURE = FIXTURE.with_name('signed-successor-worker-qualification-20261003-04')
LAUNCHED_FIXTURE = FIXTURE.with_name('signed-successor-worker-qualification-20261003-05')
EDITED_FIXTURE = FIXTURE.with_name('signed-successor-worker-qualification-20261003-07')
COMPLETED_FIXTURE = FIXTURE.with_name('signed-successor-worker-qualification-20261003-08')


class OuterResourceControls(unittest.TestCase):
    def setUp(self):
        reader = audit.source.Reader(FIXTURE, seconds=120)
        self.record = reader.json('container-execution-final.json', 4 * audit.MIB)
        self.seed = reader.json('source-setup-seed.json', 4 * audit.MIB)
        # This branch tests only the resource predicate with a synthetic exit0.
        self.record['returncode'] = 0
        self.record['container_name'] = self.record['container_absence_observation']['container_name']
        inspection = self.record['actual_container_limits']['inspection']
        inspection['Name'] = '/' + self.record['container_name']
        self.record['actual_container_limits']['inspection_stdout'] = json.dumps(inspection, sort_keys=True)

    def receive(self):
        return audit.container_limits(self.record, self.seed)

    def test_synthetic_zero_exit_keeps_actual_bound_resource_observations(self):
        value = self.receive()
        self.assertEqual(value['actual_limits']['Memory'], 8 * 1024**3)
        self.assertFalse(value['process_origin_attested'])

    def test_actual_failed_native_exit_is_rejected(self):
        self.record['returncode'] = 1
        with self.assertRaises(ValueError):self.receive()

    def test_boolean_native_exit_is_not_integer_zero(self):
        self.record['returncode'] = False
        with self.assertRaises(ValueError):self.receive()

    def test_unreleased_host_envelope_is_rejected(self):
        self.record['host_reservation_released'] = False
        with self.assertRaises(ValueError):self.receive()

    def test_container_removal_alias_cannot_replace_true(self):
        self.record['container_removed'] = 1
        with self.assertRaises(ValueError):self.receive()

    def test_raw_engine_inspection_cannot_disagree_with_parsed_record(self):
        self.record['actual_container_limits']['inspection']['HostConfig']['Memory'] = 1
        with self.assertRaises(ValueError):self.receive()

    def test_raw_cgroup_cannot_disagree_with_parsed_record(self):
        self.record['actual_container_limits']['cgroup']['values']['memory.max'] = 'max'
        with self.assertRaises(ValueError):self.receive()

    def test_engine_absence_must_name_same_container(self):
        self.record['container_absence_observation']['container_id'] = 'another'
        with self.assertRaises(ValueError):self.receive()

    def test_engine_absence_must_join_actual_inspected_container_name(self):
        self.record['container_absence_observation']['container_name'] = 'foreign'
        with self.assertRaises(ValueError):self.receive()

    def test_nonempty_live_engine_result_cannot_be_cleaned(self):
        self.record['container_absence_observation']['stdout'] = self.record['container_id'] + '\n'
        with self.assertRaises(ValueError):self.receive()

    def test_failed_engine_absence_probe_cannot_be_cleaned(self):
        self.record['container_absence_observation']['returncode'] = 1
        with self.assertRaises(ValueError):self.receive()

    def test_private_transfer_locator_is_exact(self):
        self.seed['staged_destination'] += '-other'
        with self.assertRaises(ValueError):self.receive()

    def test_concurrent_original_source_change_is_reported_beside_frozen_copies(self):
        self.record['original_selected_source_changes_after_execution'] = ['/other/work/changed.py']
        self.assertEqual(self.receive()['original_selected_source_changes_observed'], ['/other/work/changed.py'])

    def test_foreign_common_pool_lease_is_preserved_as_an_observation(self):
        self.record['host_resources_after_cleanup']['active_lease_count']=1
        self.assertEqual(self.receive()['host_global_resources_observed_after_cleanup']['active_lease_count'],1)

    def test_host_cleanup_cannot_name_another_owner(self):
        self.record['host_owned_resources_after_cleanup']['owner_pids']=[1]
        with self.assertRaises(ValueError):self.receive()

    def test_owned_host_lease_cannot_remain_after_cleanup(self):
        self.record['host_owned_resources_after_cleanup']['active_lease_count']=1
        with self.assertRaises(ValueError):self.receive()


class NativeCostControls(unittest.TestCase):
    def setUp(self):
        reader = audit.source.Reader(FIXTURE, seconds=120)
        self.result = reader.json('native/result.json', 4 * audit.MIB)
        self.result.update(qualified=True, native_worker_qualified=True,
            full_administrator_population_completed=True, native_source_generation_advanced=True)
        self.result['phases']=[{'name':name,'status':'completed','elapsed_seconds':0} for name in audit.PHASES]
        self.result['controls'] = [{'name':'published_source_rejects_old_completion','refused':True,
            'integrity_refusal_claimed':True,'error_type':'StaleCodebaseError','error':'resume catalog head changed'}]

    def receive(self):return audit.costs(self.result)

    def test_synthetic_completed_cost_branch_retains_actual_pair_timings(self):
        value = self.receive();self.assertEqual(value['new_fitting_epochs'], 0)
        self.assertLess(value['paired_close_seconds'], 30)

    def test_actual_failed_native_result_is_rejected(self):
        self.result['qualified'] = False
        with self.assertRaises(ValueError):self.receive()

    def test_fitting_boolean_alias_is_rejected(self):
        self.result['new_fitting_epochs'] = False
        with self.assertRaises(ValueError):self.receive()

    def test_new_training_cannot_be_hidden_as_inherited(self):
        self.result['new_fitting_epochs'] = 1
        with self.assertRaises(ValueError):self.receive()

    def test_new_scan_page_cannot_be_hidden_as_inherited(self):
        self.result['new_scan_pages'] = 1
        with self.assertRaises(ValueError):self.receive()

    def test_scope_does_not_gain_cuda_or_proof_authority(self):
        for name in ('proof_authority','cuda_qualified','384d_qualified'):
            changed = deepcopy(self.result);changed[name]=True
            with self.subTest(name=name),self.assertRaises(ValueError):audit.costs(changed)

    def test_nan_or_infinite_timing_is_rejected(self):
        for value in (float('nan'),float('inf')):
            changed=deepcopy(self.result);changed['paired_close_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):audit.costs(changed)

    def test_start30_deadline_is_not_relaxed(self):
        self.result['operation_deadlines']['native_start_ms'] = 60000
        with self.assertRaises(ValueError):self.receive()

    def test_timeout_is_not_stale_integrity_refusal(self):
        self.result['controls'][0]['error_type'] = 'LeaseTimeoutError'
        with self.assertRaises(ValueError):self.receive()

    def test_unknown_error_is_not_stale_integrity_refusal(self):
        self.result['controls'][0]['error_type'] = 'InventedError'
        with self.assertRaises(ValueError):self.receive()

    def test_signed_metadata_budget_does_not_relax_standalone_entry(self):
        self.result['operation_deadlines']['default_receiving'] = 180
        with self.assertRaises(ValueError):self.receive()

    def test_signed_metadata_budget_is_bounded(self):
        self.result['operation_deadlines']['admission'] = 600
        with self.assertRaises(ValueError):self.receive()

    def test_phase_population_cannot_be_empty(self):
        self.result['phases']=[]
        with self.assertRaises(ValueError):self.receive()

    def test_phase_population_cannot_omit_stop(self):
        self.result['phases']=[r for r in self.result['phases'] if r['name']!='native_worker_stop_and_uid_cleanup']
        with self.assertRaises(ValueError):self.receive()

    def test_phase_population_cannot_be_reordered(self):
        self.result['phases'].reverse()
        with self.assertRaises(ValueError):self.receive()

    def test_unknown_refusal_does_not_qualify_stale_scan(self):
        self.result['controls'][0]['name'] = 'other'
        with self.assertRaises(ValueError):self.receive()

    def test_reference_budget_failure_is_not_integrity_failure(self):
        self.result['public_receivers_reference_close']['integrity_refusal_claimed'] = True
        with self.assertRaises(ValueError):self.receive()


class ActualPublishedRefusalControls(unittest.TestCase):
    """Genuine08 cost/refusal observations only, not a whole-archive audit."""

    def setUp(self):
        self.result = audit.source.Reader(COMPLETED_FIXTURE, seconds=120).json('native/result.json', 8 * audit.MIB)

    def test_actual08_costs_accept_exact_published_head_integrity_refusal(self):
        value = audit.costs(self.result)
        self.assertEqual(value['new_fitting_epochs'], 0)
        self.assertLess(value['paired_close_seconds'], 30)

    def test_value_error_alias_does_not_authenticate_native_head_refusal(self):
        self.result['controls'][0]['error_type'] = 'ValueError'
        with self.assertRaises(ValueError):audit.costs(self.result)

    def test_timeout_cannot_become_a_stale_head_integrity_refusal(self):
        self.result['controls'][0]['error_type'] = 'LeaseTimeout'
        with self.assertRaises(ValueError):audit.costs(self.result)

    def test_unknown_error_cannot_become_a_stale_head_integrity_refusal(self):
        self.result['controls'][0]['error_type'] = 'UnknownError'
        with self.assertRaises(ValueError):audit.costs(self.result)

    def test_native_stale_class_requires_the_exact_head_changed_message(self):
        self.result['controls'][0]['error'] = 'other stale condition'
        with self.assertRaises(ValueError):audit.costs(self.result)


class ClosedArchiveFailureControls(unittest.TestCase):
    def test_real_public_context_passes_until_expected_missing_worker_logs(self):
        reader=audit.source.Reader(PREPARED_FIXTURE,seconds=120)
        prepared=reader.json('native/public-worker-context.json',4*audit.MIB)
        with self.assertRaisesRegex(ValueError,'^native private launch logs unavailable$'):
            audit.worker.verify_authored_worker_receipt(PREPARED_FIXTURE/'native',
                prepared=prepared,task_cid=prepared['task_cid'])

    def test_actual_failed_archive_stays_preserved_without_native_or_command_work(self):
        import builtins
        import os
        import subprocess
        original = builtins.__import__
        def import_only_readers(name, *args, **kwargs):
            if name.startswith(('ipfs_datasets_py', 'ipfs_accelerate_py', 'duckdb', 'torch')):
                raise AssertionError('archive reader attempted a product/native import')
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', import_only_readers), \
                patch.object(subprocess, 'run', side_effect=AssertionError('command execution forbidden')), \
                patch.object(os, 'system', side_effect=AssertionError('command execution forbidden')):
            result = audit.audit_closed_archive(FIXTURE)
        self.assertFalse(result['qualified'])
        self.assertTrue(result['preserved'])
        self.assertTrue(result['errors'])
        for name in ('native_owners_opened','sql_executed','git_executed','docker_executed','network_executed'):
            self.assertIs(result[name], False)


class LocalPoolControls(unittest.TestCase):
    def setUp(self):
        reader = audit.source.Reader(PREPARED_FIXTURE, seconds=120)
        self.authority = reader.json('native/container-resource-authority.json')
        self.boundary = reader.json('native/container-boundary.json')
        self.result = reader.json('native/result.json', 4 * audit.MIB)
        self.record = reader.json('container-execution-final.json', 4 * audit.MIB)
        # Owner-close was not reached. Supply the explicit synthetic predicate
        # fixture only; this is not a completed lifecycle or archive audit.
        self.result['resource_after_native_owner_close'] = deepcopy(self.result['final_resources'])

    def receive(self):
        return audit.local_pool(self.authority, self.record, self.result, self.boundary)

    def test_actual_cgroup_and_held_parent_with_synthetic_closed_owner(self):
        value=self.receive();self.assertFalse(value['shared_host_pid_state'])

    def test_container_cannot_attach_host_pid_accounting(self):
        self.authority['state_path']=self.record['host_scheduler_state_path']
        with self.assertRaises(ValueError):self.receive()

    def test_local_ram_cannot_exceed_actual_cgroup(self):
        self.authority['persisted_config']['total_memory_mb']=16384
        self.result['scheduler_configuration']=deepcopy(self.authority['persisted_config'])
        with self.assertRaises(ValueError):self.receive()

    def test_parent_lease_full_observation_must_join(self):
        self.authority['parent_host_envelope']['host_reservation']['owner_pid']+=1
        with self.assertRaises(ValueError):self.receive()

    def test_actual_container_namespace_cannot_be_substituted(self):
        self.authority['namespace']['pid']='pid:[other]'
        with self.assertRaises(ValueError):self.receive()

    def test_gpu_claim_cannot_be_added_to_cpu_only_pool(self):
        self.authority['persisted_config']['total_gpu_memory_mb']=1
        self.result['scheduler_configuration']=deepcopy(self.authority['persisted_config'])
        with self.assertRaises(ValueError):self.receive()

    def test_parent_cgroup_limit_cannot_be_substituted(self):
        self.authority['parent_host_envelope']['memory_limit_bytes']=1
        with self.assertRaises(ValueError):self.receive()

    def test_extended_authority_key_is_rejected(self):
        self.authority['currentness_authority']=True
        with self.assertRaises(ValueError):self.receive()

    def test_container_local_other_participant_must_close(self):
        self.result['final_resources']['global_active_lease_count']=1
        with self.assertRaises(ValueError):self.receive()


class PublicCheckControls(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.bodies={'check_type.py':b'authored synthetic type source\n',
                     'check_offset.py':b'authored synthetic offset source\n'}
        self.rows=[]
        for name,raw in self.bodies.items():
            source={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            row={'schema':'source-successor-published-public-check@1','path':name,
                'argv':['/usr/local/bin/python','-B',name],'cwd':'/results/native/repository',
                'timeout_seconds':10,'returncode':0,'source_before':source,'source_after':deepcopy(source)}
            for role,value in [('stdout','synthetic passed\n'),('stderr','')]:
                row[role]=value;row[role+'_bytes']=len(value.encode());row[role+'_sha256']=hashlib.sha256(value.encode()).hexdigest()
            self.rows.append(row)
        owner=self
        class Reader:
            def raw(self,path,limit):
                if not path.startswith('native/repository/'):raise AssertionError('unexpected source locator')
                return owner.bodies[path.removeprefix('native/repository/')]
        self.reader=Reader()

    def receive(self):return audit.verify_public_checks(self.rows,self.reader)

    def test_synthetic_two_check_byte_join_only(self):self.assertEqual(len(self.receive()),2)

    def test_missing_original_check_is_rejected(self):
        self.rows.pop()
        with self.assertRaises(ValueError):self.receive()

    def test_check_source_cannot_change_after_receipt(self):
        self.bodies['check_offset.py']+=b'changed'
        with self.assertRaises(ValueError):self.receive()

    def test_actual_output_pin_cannot_disagree(self):
        self.rows[0]['stdout']+='changed'
        with self.assertRaises(ValueError):self.receive()

    def test_source_mutation_during_check_is_rejected(self):
        self.rows[0]['source_before']['sha256']='0'*64
        with self.assertRaises(ValueError):self.receive()

    def test_failed_check_cannot_be_published(self):
        self.rows[0]['returncode']=1
        with self.assertRaises(ValueError):self.receive()

    def test_boolean_check_exit_is_rejected(self):
        self.rows[0]['returncode']=False
        with self.assertRaises(ValueError):self.receive()

    def test_checker_path_must_join_invocation(self):
        self.rows[0]['path']='check_other.py'
        with self.assertRaises(ValueError):self.receive()

    def test_original_check_timeout_is_retained(self):
        self.rows[0]['timeout_seconds']=300
        with self.assertRaises(ValueError):self.receive()

    def test_combined_output_bound_cannot_be_split_between_streams(self):
        import hashlib
        for role in ('stdout','stderr'):
            raw=b'x'*33000;self.rows[0][role]=raw.decode()
            self.rows[0][role+'_bytes']=len(raw);self.rows[0][role+'_sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):self.receive()


class StagedCustodyControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=audit.source.Reader(FIXTURE,seconds=120)
        cls.baseline=cls.reader.whole_archive()
        cls.seed=cls.reader.json('source-setup-seed.json',4*audit.MIB)
        cls.record=cls.reader.json('container-execution-final.json',4*audit.MIB)

    def setUp(self):self.archive=deepcopy(self.baseline)
    def receive(self):return audit.verify_staged_custody(self.reader,self.archive,self.seed,self.record)

    def test_actual_closed_failed_seed_byte_custody_only(self):
        self.assertEqual(self.receive()['staged_files'],1634)

    def test_missing_staged_member_is_rejected(self):
        self.archive['files']=[r for r in self.archive['files'] if r['path']!='setup-seed/repository/calc.py']
        with self.assertRaises(ValueError):self.receive()

    def test_changed_staged_source_mode_is_rejected(self):
        next(r for r in self.archive['files'] if r['path']=='setup-seed/repository/calc.py')['mode']^=0o100
        with self.assertRaises(ValueError):self.receive()

    def test_changed_staged_source_pin_is_rejected(self):
        next(r for r in self.archive['files'] if r['path']=='setup-seed/repository/calc.py')['sha256']='0'*64
        with self.assertRaises(ValueError):self.receive()

    def test_unlisted_staged_member_is_rejected(self):
        self.archive['files'].append({'path':'setup-seed/extra','kind':'file','bytes':1,'sha256':'0'*64,'mode':0o444})
        with self.assertRaises(ValueError):self.receive()


class PreservedRepositoryControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader = audit.source.Reader(LAUNCHED_FIXTURE, seconds=120)
        cls.baseline = reader.whole_archive()
        cls.seed = reader.json('source-setup-seed.json', 4 * audit.MIB)
        cls.prior = reader.json('native/closed-full-scan/seed-evidence/current-manifest.json', 4 * audit.MIB)['value']

    def setUp(self):
        self.archive = deepcopy(self.baseline)

    def receive(self):
        return audit.verify_preserved_repository(self.archive, self.seed, self.prior)

    def opaque(self):
        return next(row for row in self.archive['files'] if row['path'] == 'native/repository/oversized.dat')

    def test_actual_failed_archive_pins_299_other_files_including_opaque_bytes(self):
        value = self.receive()
        self.assertEqual(value['unchanged_repository_files'], 299)
        self.assertEqual(value['opaque_files_byte_pinned'], 2)
        self.assertFalse(value['source_semantics_verified'])

    def test_opaque_file_cannot_be_missing(self):
        self.archive['files'].remove(self.opaque())
        with self.assertRaises(ValueError): self.receive()

    def test_opaque_bytes_cannot_change_under_unchanged_capture_identity(self):
        self.opaque()['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.receive()

    def test_opaque_size_cannot_change(self):
        self.opaque()['bytes'] += 1
        with self.assertRaises(ValueError): self.receive()

    def test_opaque_mode_cannot_change(self):
        self.opaque()['mode'] ^= 0o100
        with self.assertRaises(ValueError): self.receive()


class CompleteScanCustodyControls(unittest.TestCase):
    def test_inherited_ten_page_scan_and_two_checkpoint_states_receive_without_owners(self):
        reader=audit.source.Reader(FIXTURE,seconds=120)
        seed=reader.json('source-setup-seed.json',4*audit.MIB)
        result=reader.json('native/result.json',4*audit.MIB)
        states=reader.json('native/checkpoint-states-before.json',4*audit.MIB)
        observed=audit.verify_complete_scan(reader,seed,result,states)
        self.assertEqual(observed['complete_scan']['coverage']['inventory_entries'],300)
        self.assertEqual(observed['complete_scan']['coverage']['pages'],10)
        self.assertFalse(observed['fresh_native_execution_reperformed'])


class ExecutionScopeJoinControls(unittest.TestCase):
    def test_actual_prelaunch_signed_scope_joins_failed_worker_without_qualifying_lifecycle(self):
        reader=audit.source.Reader(LAUNCHED_FIXTURE,seconds=120)
        envelope=reader.json('native/execution-scope-before.json',16*audit.MIB)
        seed=reader.json('source-setup-seed.json',4*audit.MIB)
        prepared=reader.json('native/public-worker-context.json',4*audit.MIB)
        value=audit.verify_execution_scope(audit.worker.signature(envelope),envelope['binding'],reader,seed,prepared)
        self.assertTrue(value['current_admission_joined'])
        self.assertFalse(value['process_origin_attested'])

    def setUp(self):
        import hashlib
        import shlex
        self.reader=audit.source.Reader(PREPARED_FIXTURE,seconds=120)
        self.seed=self.reader.json('source-setup-seed.json',4*audit.MIB)
        self.prepared=self.reader.json('native/public-worker-context.json',4*audit.MIB)
        admission=self.reader.json('native/admission.json',16*audit.MIB)
        candidate=self.reader.json('native/candidate.json',4*audit.MIB)
        # Unsigned scope predicate only. No actual scope existed in failed04.
        self.scope={'schema':'supervisor-codebase-inventory-execution-scope@1',
            'profile':'codebase-inventory-one-ready-native-worker@1','admission_cid':audit.worker.structured(admission),
            'current_inventory':self.reader.json('native/current-admission-verification.json',4*audit.MIB),
            'head':self.seed['current_head'],'candidate':candidate|{'implementation_command':shlex.join(candidate['argv'])},
            'implementation':{name:hashlib.sha256(self.reader.raw(
                ('datasets' if name.startswith('ipfs_datasets_py.') else 'source')+'/'+name.replace('.','/')+'.py',4*audit.MIB)).hexdigest()
                for name in audit.SCOPE_MODULES}}
        self.binding=deepcopy(admission['manifest']['binding'])

    def receive(self):return audit.verify_execution_scope(self.scope,self.binding,self.reader,self.seed,self.prepared)

    def test_unsigned_predicate_joins_actual_admission_and_thirteen_source_bodies(self):
        self.assertEqual(self.receive()['listed_execution_producers'],13)

    def test_scope_cannot_select_another_admission(self):
        self.scope['admission_cid']=self.seed['root_cid']
        with self.assertRaises(ValueError):self.receive()

    def test_scope_cannot_select_another_head(self):
        self.scope['head']=self.seed['previous_head']
        with self.assertRaises(ValueError):self.receive()

    def test_scope_cannot_select_another_author(self):
        self.binding['identity']='did:key:other'
        with self.assertRaises(ValueError):self.receive()

    def test_scope_command_must_join_literal_arguments(self):
        self.scope['candidate']['implementation_command']+=' changed'
        with self.assertRaises(ValueError):self.receive()

    def test_signed_candidate_cannot_change_ready_revision(self):
        self.scope['candidate']['task_revision']+=1
        with self.assertRaises(ValueError):self.receive()

    def test_missing_execution_producer_is_rejected(self):
        self.scope['implementation'].pop(next(iter(self.scope['implementation'])))
        with self.assertRaises(ValueError):self.receive()

    def test_changed_execution_producer_pin_is_rejected(self):
        self.scope['implementation'][next(iter(self.scope['implementation']))]='0'*64
        with self.assertRaises(ValueError):self.receive()


class FixtureCleanupControls(unittest.TestCase):
    def setUp(self):
        # Genuine failed07 translation preimages, with a synthetic successful
        # cleanup decision only. Neither this predicate nor its positive case
        # qualifies07, asserts physical removal, or opens a native owner.
        original = audit.source.Reader(EDITED_FIXTURE, seconds=120)
        self.task = original.json('native/public-worker-context.json')['task_cid']
        self.population = original.json('native/execution-scope-before.json', 8 * audit.MIB)['payload']['native_population']
        self.allocations = original.json('native/result.json', 8 * audit.MIB)['observed_worker_allocations']
        self.allocation = self.allocations[1]
        state = 'native/' + Path(self.allocation['state_dir']).relative_to('/results/native').as_posix()
        self.state = state
        raw = {state + '/' + name: original.raw(state + '/' + name, 4 * audit.MIB) for name in
            ('database-attempt-binding.json','task-projection.runtime.todo.md','task_queue.json')}
        class RetainedBytes:
            def raw(self, path, maximum):
                value = raw[path]
                if len(value) > maximum:raise ValueError('test receiving byte limit')
                return value
        self.raw = raw
        self.reader = RetainedBytes()
        self.rows=[{'allocation':deepcopy(self.allocation),'scope':'explicit owner fixture cleanup after native STOP',
            'decision':{'allowed':True,'disposition':'allow','reason':'synthetic','failure_kind':'implementation',
                'provider_call_allowed':True,'attempt_consumed':True,'record':deepcopy(self.allocation)}}]
        self.archive={'files':[]}

    def receive(self):return audit.verify_fixture_cleanup(self.rows,self.allocations,self.archive,self.task,
        reader=self.reader,population=self.population)

    def identity(self):return audit.verify_portal_cleanup_identity(self.reader,self.allocation,self.task,self.population)

    def mutate_json(self, name, callback, *, rehash=False):
        path=self.state+'/'+name;value=audit.source.parse(self.raw[path]);callback(value)
        if rehash:
            value['binding_id']='sha256:'+audit.hashlib.sha256(audit.worker.wire(
                {key:item for key,item in value.items() if key!='binding_id'})).hexdigest()
        self.raw[path]=audit.source.numerical_wire(value)

    def mutate_projection(self, before, after):
        path=self.state+'/task-projection.runtime.todo.md'
        self.assertIn(before,self.raw[path]);self.raw[path]=self.raw[path].replace(before,after,1)

    def test_synthetic_fixture_cleanup_join_only(self):
        self.assertEqual(self.receive()['explicit_fixture_allocations_removed'],1)

    def test_actual07_portal_identity_replays_exact_signed_intent_translation(self):
        value=self.identity()
        self.assertNotEqual(value['portal_task_cid'],self.task)
        self.assertEqual(value['portal_task_cid'],self.allocation['canonical_task_cid'])
        self.assertFalse(value['projection_authority'])
        self.assertFalse(value['control_task_projection_replayed'])
        self.assertFalse(value['native_persistence_requeried'])

    def test_superseded_preparing_observation_is_not_counted_as_removed(self):
        value=self.receive()
        self.assertEqual(value['explicit_fixture_allocations_removed'],1)
        self.assertEqual(value['preparing_observations_without_removal_receipt'],1)
        self.assertFalse(value['preparing_physical_absence_verified'])

    def test_completed_mutable_status_preserves_exact_seed_and_immutable_binding(self):
        self.mutate_projection(b'- Status: ready\n',b'- Status: completed\n')
        self.assertEqual(self.identity()['portal_task_cid'],self.allocation['canonical_task_cid'])

    def test_queue_alias_alone_cannot_replace_rebuilt_canonical_identity(self):
        def change(value):
            old=self.allocation['canonical_task_cid'];row=value['entries'].pop(old)
            row['canonical_task_cid']=self.task;value['entries'][self.task]=row
            value['aliases']={key:self.task for key in value['aliases']}
        self.mutate_json('task_queue.json',change)
        with self.assertRaises(ValueError):self.identity()

    def test_queue_semantic_key_must_match_signed_task_preimage(self):
        self.mutate_json('task_queue.json',lambda value:value['entries'][self.allocation['canonical_task_cid']].update(
            canonical_task_key='task/v1/'+'0'*64))
        with self.assertRaises(ValueError):self.identity()

    def test_queue_provenance_cannot_select_another_attempt_projection(self):
        def change(value):value['entries'][self.allocation['canonical_task_cid']]['provenance'][0]['source_path']+='.foreign'
        self.mutate_json('task_queue.json',change)
        with self.assertRaises(ValueError):self.identity()

    def test_rehashed_binding_cannot_select_another_signed_task(self):
        self.mutate_json('database-attempt-binding.json',lambda value:value.update(task_cid='foreign'),rehash=True)
        with self.assertRaises(ValueError):self.identity()

    def test_binding_authority_false_rejects_integer_alias(self):
        self.mutate_json('database-attempt-binding.json',lambda value:value.update(projection_authority=0),rehash=True)
        with self.assertRaises(ValueError):self.identity()

    def test_rehashed_boolean_revision_is_not_an_admitted_claim_revision(self):
        self.mutate_json('database-attempt-binding.json',lambda value:value.update(task_revision=True),rehash=True)
        with self.assertRaises(ValueError):self.identity()

    def test_rehashed_model_independent_claim_body_digest_cannot_be_substituted(self):
        self.mutate_json('database-attempt-binding.json',lambda value:value.update(task_body_digest='sha256:'+'0'*64),rehash=True)
        with self.assertRaises(ValueError):self.identity()

    def test_rehashed_control_binding_cannot_change_portal_basis(self):
        self.mutate_json('database-attempt-binding.json',lambda value:value.update(control_portal_binding_basis_cid=self.task),rehash=True)
        with self.assertRaises(ValueError):self.identity()

    def test_duplicate_projection_field_is_refused(self):
        path=self.state+'/task-projection.runtime.todo.md';self.raw[path]+=b'- Database task CID: foreign\n'
        with self.assertRaises(ValueError):self.identity()

    def test_projection_validation_must_match_original_signed_native_task(self):
        self.mutate_projection(b'python3 -B check_offset.py',b'python3 -B check_type.py')
        with self.assertRaises(ValueError):self.identity()

    def test_projection_output_cannot_expand_original_permission(self):
        self.mutate_projection(b'- Outputs: calc.py',b'- Outputs: calc.py, foreign.py')
        with self.assertRaises(ValueError):self.identity()

    def test_projection_original_signed_contract_cid_is_exact(self):
        self.mutate_projection(b'- Local planning contract CID: baguqeera',b'- Local planning contract CID: foreignbaguqeera')
        with self.assertRaises(ValueError):self.identity()

    def test_attempt_directory_is_bound_to_exact_attempt_id(self):
        self.allocation['state_dir']='/results/native/private/launch/state/run/admitted_database_portal_attempts/'+'0'*24
        with self.assertRaises((KeyError,ValueError)):self.identity()

    def test_packed_lifecycle_attempt_cannot_be_reused_for_another_attempt(self):
        self.allocation['attempt']+=1
        with self.assertRaises(ValueError):self.identity()

    def test_lifecycle_record_cid_must_bind_exact_workspace(self):
        self.allocation['record_id']='foreign'
        with self.assertRaises(ValueError):self.identity()

    def test_skipped_active_allocation_cannot_be_called_preparing(self):
        self.allocations[0]['state']='active'
        with self.assertRaises(ValueError):self.receive()

    def test_skipped_preparing_observation_requires_same_worktree_lease(self):
        self.allocations[0]['lease_id']='foreign'
        with self.assertRaises(ValueError):self.receive()

    def test_skipped_preparing_fence_must_precede_removed_allocation(self):
        self.allocations[0]['fence']=self.allocation['fence']
        with self.assertRaises(ValueError):self.receive()

    def test_missing_cleanup_is_rejected(self):
        self.rows=[]
        with self.assertRaises(ValueError):self.receive()

    def test_native_disallowed_cleanup_is_rejected(self):
        self.rows[0]['decision']['allowed']=False
        with self.assertRaises(ValueError):self.receive()

    def test_boolean_allowed_alias_is_rejected(self):
        self.rows[0]['decision']['allowed']=1
        with self.assertRaises(ValueError):self.receive()

    def test_cleanup_identity_must_join_observed_owner(self):
        self.rows[0]['decision']['record']['owner']['pid']+=1
        with self.assertRaises(ValueError):self.receive()

    def test_cleanup_cannot_target_other_task(self):
        self.task='other-task'
        with self.assertRaises(ValueError):self.receive()

    def test_linked_git_worktree_metadata_must_be_absent(self):
        self.archive['files'].append({'path':'native/repository/.git/worktrees/fixture'})
        with self.assertRaises(ValueError):self.receive()

    def test_allocation_cannot_escape_owned_root(self):
        self.allocation['workspace_path']='/foreign/worktree'
        self.rows[0]['allocation']=deepcopy(self.allocation)
        with self.assertRaises(ValueError):self.receive()


class NativePopulationControls(unittest.TestCase):
    def setUp(self):
        self.reader=audit.source.Reader(LAUNCHED_FIXTURE,seconds=120)
        envelope=self.reader.json('native/execution-scope-before.json',16*audit.MIB)
        self.population=deepcopy(envelope['payload']['native_population'])
        self.admission=self.reader.json('native/admission.json',16*audit.MIB)
        self.prepared=self.reader.json('native/public-worker-context.json',4*audit.MIB)

    def receive(self):return audit.verify_native_population(self.population,self.admission,self.reader,self.prepared)

    def test_actual_signed_prelaunch_population_only_not_failed_worker_completion(self):
        self.assertEqual(self.receive()['original_signed_tasks'],2)

    def test_original_task_cannot_be_omitted(self):
        self.population['tasks'].pop()
        with self.assertRaises(ValueError):self.receive()

    def test_original_dependency_cannot_be_removed(self):
        next(r for r in self.population['tasks'] if r['task_alias']=='SUCCESSOR-FORMAT')['dependencies']=[]
        with self.assertRaises(ValueError):self.receive()

    def test_other_task_cannot_be_selected(self):
        self.population['selected_task_cids']=[next(r['task_cid'] for r in self.population['tasks'] if r['task_alias']=='SUCCESSOR-TYPE')]
        with self.assertRaises(ValueError):self.receive()

    def test_fake_prerequisite_ready_status_is_rejected(self):
        next(r for r in self.population['tasks'] if r['task_alias']=='SUCCESSOR-TYPE')['status']='ready'
        with self.assertRaises(ValueError):self.receive()

    def test_boolean_native_revision_is_rejected(self):
        next(r for r in self.population['tasks'] if r['task_alias']=='SUCCESSOR-FORMAT')['revision']=True
        with self.assertRaises(ValueError):self.receive()

    def test_prerequisite_receipt_evidence_cannot_change(self):
        next(iter(self.population['completion_rows'].values()))[0][8]=self.prepared['task_cid']
        with self.assertRaises(ValueError):self.receive()

    def test_native_validation_cannot_be_removed_while_signed_contract_remains(self):
        self.population['tasks'][0]['validations']=[]
        with self.assertRaises(ValueError):self.receive()

    def test_native_output_permission_cannot_expand(self):
        self.population['tasks'][0]['outputs'][0]['path']='other.py'
        with self.assertRaises(ValueError):self.receive()


if __name__ == '__main__':unittest.main(verbosity=2)
