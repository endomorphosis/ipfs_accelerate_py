"""Finite, source-bound service around the existing original native owner.

Only closed, genuine idle observations may poll. Unknown or failed cycles leave
a persistent hold and exit 78; restarting this service never retries them.
The original owner retains all compute, storage and task execution authority.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

CONFIG_SCHEMA = 'agent-supervisor/scientific-executor-service@1'
TEMPLATE_SCHEMA = 'agent-supervisor/scientific-executor-cycle-template@1'
CONFIG_KEYS = frozenset({'schema', 'profile_path', 'catalogue', 'idle_template_ref',
    'execution_templates', 'poll_seconds', 'max_cycles', 'max_proof_bytes'})
CATALOGUE_KEYS = frozenset({'database_path', 'population_path', 'board_id',
    'board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256',
    'max_age_seconds', 'timeout_seconds'})
TEMPLATE_KEYS = frozenset({'schema', 'kind', 'seed_root', 'request_ref', 'plan_ref', 'links'})
HOLD_EXIT = 78
QUIET_STATUSES = frozenset({'blocked', 'completed', 'complete', 'done', 'skipped'})
RECOVERY_SCHEMA = 'agent-supervisor/scientific-known-nonstarted-recovery@1'
RECOVERY_SHA_ENV = 'IPFS_SCIENTIFIC_KNOWN_NONSTARTED_RECOVERY_SHA256'
RECOVERY_KEYS = frozenset({'schema', 'profile_ref', 'prior_profile_ref', 'prior_profile_snapshot_ref',
    'activation_capture_ref', 'activation_plan_ref', 'negative_audit_ref', 'memory_correction_ref',
    'source_commitment_ref', 'hold_ref', 'intent_ref', 'request_ref', 'owner_stdout_ref',
    'failed_cycle_root', 'scheduler_snapshot_ref', 'max_cycles', 'authorization', 'recorded_at', 'manager_before'})
RECOVERY_ARCHIVE_FIELDS = ('prior_profile_snapshot_ref', 'activation_capture_ref', 'activation_plan_ref',
    'negative_audit_ref', 'memory_correction_ref', 'source_commitment_ref', 'scheduler_snapshot_ref')


CLOSED_IDLE_SCHEMA = 'agent-supervisor/scientific-known-closed-idle-recovery@1'
CLOSED_IDLE_SHA_ENV = 'IPFS_SCIENTIFIC_KNOWN_CLOSED_IDLE_RECOVERY_SHA256'
CLOSED_IDLE_KEYS = frozenset({'schema', 'recovery_kind', 'profile_ref', 'prior_profile_ref',
    'prior_profile_snapshot_ref', 'prior_authorization_ref', 'prior_ack_ref',
    'prior_administrative_capture_ref', 'negative_audit_ref', 'source_commitment_ref',
    'scheduler_snapshot_ref', 'current_native_ref', 'global_hold_ref', 'cycle_hold_ref',
    'cycle_proof_ref', 'max_cycles', 'authorization', 'recorded_at', 'manager_before'})
CLOSED_IDLE_ARCHIVE_FIELDS = ('prior_profile_snapshot_ref', 'prior_administrative_capture_ref',
    'negative_audit_ref', 'source_commitment_ref', 'scheduler_snapshot_ref', 'current_native_ref')


GENERATION_SCHEMA = 'agent-supervisor/scientific-bounded-generation-handover@1'
GENERATION_SHA_ENV = 'IPFS_SCIENTIFIC_BOUNDED_GENERATION_SHA256'
GENERATION_KEYS = frozenset({'schema','authorization','profile_ref','prior_profile_ref','prior_profile_snapshot_ref',
    'prior_config_ref','prior_config_snapshot_ref','retained_census_ref','prior_global_hold_ref','prior_cycle_hold_ref',
    'prior_intent_ref','prior_last_closed_cycle_ref','prior_main_identity','source_capsule_ref',
    'cache_initialization_request_ref','cache_initialization_terminal_ref','scheduler_snapshot_ref',
    'retained_generations','generation_number','max_generations','global_max_bytes','spent_cycle_count',
    'recorded_at','manager_before'})
ORIGINAL_NAMED_ROOTS = (Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs'),
    Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues'),
    Path('/home/barberb/lift_coding/external/ipfs_datasets/docs/implementation/reports/evidence'), Path('/tmp/pytest-of-barberb'))
ORIGINAL_RESOURCE_LEDGER = ORIGINAL_NAMED_ROOTS[0]/'federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
GLOBAL_RETAINED_BYTES = 220000000000
# Original native storage controls the entire attempt. The producer writes at
# most twelve bounded JSON/stdout artifacts outside that attempt. The v3 actor
# enforces <=256 created file paths and <=32 process constructors, while its
# original parent enforces <=276 total attempt files. Twelve producer files
# outside that attempt complete the fixed288-path allowance.
REUSE_CYCLE_RESERVE_BYTES = 384 * 1024**2 + 12 * 4 * 1024**2
REUSE_CYCLE_RESERVE_FILES = 256 + 32


class ExecutorHold(ValueError):
    """A cycle cannot be replayed or silently discarded."""


ADMISSION_SCHEMA = 'agent-supervisor/scientific-source-admission-recovery@1'
ADMISSION_ACK_SCHEMA = 'agent-supervisor/scientific-source-admission-reconciliation@1'
ADMISSION_SHA_ENV = 'IPFS_SCIENTIFIC_SOURCE_ADMISSION_RECOVERY_SHA256'
MAX_CONSECUTIVE_SOURCE_ADMISSION_DEFERRALS = 3
ADMISSION_KEYS = frozenset({'schema','authorization','profile_ref','prior_profile_ref','prior_profile_snapshot_ref',
    'prior_config_snapshot_ref','prior_main_identity','prior_generation_authorization_ref','prior_generation_ack_ref',
    'prior_health_receipt_ref','global_hold_ref','cycle_hold_ref','failed_intent_ref','failed_request_ref','failed_stdout_ref',
    'negative_audit_ref','scheduler_snapshot_ref','current_native_ref','source_commitment_ref','spent_cycle_count',
    'generation_number','max_generations','max_cycles','max_proof_bytes','recorded_at','manager_before'})


class SourceAdmissionDeferred(ExecutorHold):
    """An immutable negative; this carries no health or execution credit."""


def source_admission_PID_absent(pid):
    """Only an actual missing proc entry means absent; unreadable is UNKNOWN."""
    try: (Path('/proc')/str(pid)).lstat()
    except (FileNotFoundError,ProcessLookupError):return True
    except OSError as error:raise ExecutorHold('source_admission_PID_observation_unknown')from error
    return False


def source_admission_terminal(terminal, request_ref, request, cycle_root, *, captured_owner=None):
    """Recognize only the original closed pre-resource source-admission timeout.

    A historical terminal without owner_birth requires literal PID absence on
    the same boot. A newly running caller can additionally join its captured
    live original-owner identity; this never fabricates a missing terminal field.
    """
    h=health_owner();root=Path(cycle_root)
    need(type(terminal)is dict and terminal.get('schema')=='legal-ir-original-native-control-terminal/v1'
         and terminal.get('request_sha256')==request_ref['sha256'] and terminal.get('completed')is False
         and terminal.get('error_type')=='SourceLeaseTimeout'
         and terminal.get('error')=='timed out waiting for admission source lease'
         and terminal.get('control_effects')=='not_started' and terminal.get('storage_status')=='not_requested'
         and terminal.get('child_launch_attempted')is False and terminal.get('child_reaped')is False
         and terminal.get('child_returncode')is None and terminal.get('source_READ_closed')is True
         and terminal.get('original_native_compute_closed')is True
         and terminal.get('absolute_resource_operation_deadline_closed')is True
         and terminal.get('task_completion_authority_granted_by_resource_owner')is False
         and terminal.get('cleanup_errors')==[] and terminal.get('received_signals')==[],
         'source_admission_terminal_not_exact_closed_negative')
    need(not any(key in terminal for key in ('reservation','execution_request','process_spawn_receipts','sampled_process_custody_final','sampled_process_custody','external_precharge','external_before','external_after','external_output_parent_identities','parent_capacity_handoff','database_preimage_before_grants','database_preimage_observed','database_postimage','allowed_subprocess_events')),
         'source_admission_terminal_has_resource_or_process_effects')
    need(request.get('schema')=='legal-ir-original-native-control-request/v1'and request.get('purpose')=='native_control_metadata'
         and request.get('request_id')=='scientific-executor:'+root.name
         and request_ref['path']==str(root/'request.json')
         and h._integer(terminal.get('owner_pid'),2) and h._text(terminal.get('owner_boot_id')),
         'source_admission_request_or_owner_identity')
    need(all(type(request.get(k))is int and request[k]==v for k,v in dict(cpu_slots=1,memory_mb=1536,child_process_slots=8,storage_bytes=384*1024**2,timeout_seconds=300).items())
         and request.get('resource_ledger')==str(ORIGINAL_RESOURCE_LEDGER),'source_admission_original_native_envelope_changed')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(terminal['owner_boot_id']==boot,'source_admission_owner_boot_changed')
    attempt=h._path(request['attempt_directory'])
    need(root in attempt.parents and (not os.path.lexists(attempt)or
         stat.S_ISDIR(attempt.lstat().st_mode)and not attempt.is_symlink()and attempt.lstat().st_uid==os.geteuid()
         and stat.S_IMODE(attempt.lstat().st_mode)==0o700 and not any(attempt.iterdir())),
         'source_admission_attempt_not_absent_or_empty')
    need(not any((root/name).exists()for name in ('terminal.json','cycle-proof.json','closed.json','native-diagnostic.json')),
         'source_admission_has_positive_or_native_evidence')
    if captured_owner is None:
        # No birth is invented or inferred for the actual historical terminal.
        need(source_admission_PID_absent(terminal['owner_pid']),'source_admission_birthless_owner_PID_not_absent')
    else:
        need(captured_owner.get('pid')==terminal['owner_pid']and captured_owner.get('boot_id')==boot
             and h._integer(captured_owner.get('birth_ticks'),1)
             and h._definitely_dead_identity(captured_owner,dict(boot_id=boot)),
             'source_admission_captured_owner_not_joined_or_still_live')
        if 'owner_birth'in terminal:
            need(terminal['owner_birth']==str(captured_owner['birth_ticks']),'source_admission_original_birth_differs')
    return terminal


def source_admission_deferral(profile, request_ref, request, root, kind, process, cycle_fields, returncode):
    """Retain one proved nonstarted negative after the genuine original wait."""
    h=health_owner()
    need(kind=='genuine_scoped_idle'and type(returncode)is int and returncode!=0
         and type(process)is subprocess.Popen and process.returncode==returncode and process.poll()is not None,
         'source_admission_deferral_requires_original_reaped_idle_owner')
    terminal=original_terminal(h._raw(root/'owner.stdout'),request_ref)
    source_admission_terminal(terminal,request_ref,request,root,captured_owner=cycle_fields['owner_observation'])
    need(cycle_fields['owner_observation']['argv']==[profile['unit']['argv'][0],'-B',profile['owner_source']['path'],
         '--request',request_ref['path'],'--expected-request-sha256',request_ref['sha256']]
         and request['source_sha256'].get(profile['owner_source']['path'])==profile['owner_source']['sha256']
         and request['source_sha256'].get(profile['driver_source']['path'])==profile['driver_source']['sha256'],
         'source_admission_original_owner_source_argv')
    negative=put(root/'admission-terminal-negative.json',terminal)
    return put(root/'admission-deferred.json',dict(schema='agent-supervisor/scientific-source-admission-deferred@1',
        request_ref=request_ref,terminal_ref=negative,original_wait_returncode=returncode,
        sequence=cycle_fields['sequence'],captured_owner=cycle_fields['owner_observation'],
        process=cycle_fields['process'],recorded_at=time.time(),health_sequence_advanced=False,
        health_or_task_completion_authority=False,automatic_retry_of_original_request=False))




def source_admission_unit_projection(argv, entrypoint):
    """Only producer/config pointers and the one explicit recovery mode may move."""
    need(type(argv)is list and argv.count(entrypoint)==1 and argv.count('--config')==argv.count('--config-sha256')==1,'source_admission_unit_argv_shape')
    result=list(argv);result[result.index(entrypoint)]='<producer>'
    for flag in ('--config','--config-sha256'):result[result.index(flag)+1]='<config>'
    flags=[flag for flag in ('--bounded-generation-handover','--known-source-admission-recovery')if flag in result]
    need(len(flags)==1 and result.count(flags[0])==1,'source_admission_unit_recovery_mode')
    index=result.index(flags[0]);need(index+1<len(result),'source_admission_unit_recovery_path');del result[index:index+2]
    return result


def source_admission_source_inventory(reference, profile):
    """Reuse bounded committed-inventory joins for producer-only source repair.

    Native owner/driver/cache may retain the already-qualified H root. Every
    selected Python source under the new producer root, including real package
    initializers, must be inventory-bound; no SDK cache generation is changed.
    """
    h=health_owner();source=h._document(reference);producer=Path(profile['main_entrypoint']['path']).parents[3]
    need(source.get('schema')=='legal-ir-scientific-source-commitment/v1'and source.get('runtime_activation_or_task_authority')is False
         and source.get('source_root')==str(producer),'source_admission_source_commitment')
    invref=source['source_inventory'];need(any(root in Path(invref['path']).parents for root in ORIGINAL_NAMED_ROOTS),'source_admission_inventory_not_durable')
    inventory=h._document(invref);rows=inventory.get('files')
    need(inventory.get('schema')=='legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(k)==source.get(k)for k in ('source_root','source_commit','source_tree'))
         and all(type(source.get(k))is str and len(source[k])==40 and set(source[k])<=set('0123456789abcdef')for k in ('source_commit','source_tree'))
         and type(rows)is list and 0<len(rows)<=4096 and inventory.get('file_count')==len(rows)
         and all(type(row)is dict and type(row.get('path'))is str for row in rows),'source_admission_inventory_shape')
    indexed={row['path']:row for row in rows};selected=[pin for pin in profile['source_descriptors']if Path(pin['path']).is_relative_to(producer)]
    parents=[producer/'ipfs_accelerate_py/__init__.py',producer/'ipfs_accelerate_py/agent_supervisor/__init__.py',producer/'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py']
    need(len(indexed)==len(rows)and len(selected)==4 and profile['main_entrypoint']in selected
         and {pin['path']for pin in selected}=={profile['main_entrypoint']['path'],*(str(path)for path in parents)}
         and all(all(indexed.get(pin['path'],{}).get(k)==pin[k]for k in ('path','bytes','sha256'))for pin in selected),
         'source_admission_producer_or_parent_source_inventory_changed')
    return inventory


# SOURCE PROPOSAL ONLY. Added to the existing scientific executor owner.
CATALOGUE_RACE_SCHEMA = 'agent-supervisor/scientific-closed-catalogue-race-recovery@1'
CATALOGUE_RACE_ACK_SCHEMA = 'agent-supervisor/scientific-closed-catalogue-race-reconciliation@1'
CATALOGUE_RACE_SHA_ENV = 'IPFS_SCIENTIFIC_CLOSED_CATALOGUE_RACE_RECOVERY_SHA256'
CATALOGUE_RACE_NEGATIVE_SCHEMA = 'agent-supervisor/scientific-closed-catalogue-race-deferred@1'
MAX_CONSECUTIVE_CATALOGUE_RACE_DEFERRALS = 3
CATALOGUE_RACE_REJECTIONS = frozenset({'embedded_idle_catalogue_cursor_changed',
    'embedded_idle_catalogue_tasks_changed'})
CATALOGUE_RACE_KEYS = frozenset({'schema','authorization','profile_ref','prior_profile_ref',
    'prior_profile_snapshot_ref','prior_config_snapshot_ref','prior_main_identity',
    'prior_generation_authorization_ref','prior_generation_ack_ref',
    'prior_source_admission_authorization_ref','prior_source_admission_ack_ref',
    'prior_health_receipt_ref','global_hold_ref','failed_cycle_hold_ref','failed_cycle_proof_ref',
    'local_cycle_census_refs','negative_audit_ref','external_progress_ref',
    'population_boundary_ref','population_boundary_independent_review_ref','current_native_ref','source_commitment_ref','scheduler_snapshot_ref',
    'native_owner_source_commitment_ref','scheduler_policy_transition_ref',
    'spent_cycle_count','generation_number','max_generations','max_cycles','max_proof_bytes',
    'recorded_at','manager_before'})


class CatalogueRaceDeferred(ExecutorHold):
    """One genuinely closed negative observation; never a health publication."""


def _catalogue_race_task_rows(rows, scope, reason):
    h=health_owner()
    need(type(rows)is list and len(rows)==len(scope) and len(scope)==len(set(scope)),reason)
    result={}
    for row in rows:
        h._keys(row,{'task_cid','status','revision'},reason)
        need(row['task_cid']in scope and row['task_cid']not in result
             and row['status']in QUIET_STATUSES and h._integer(row['revision'],1),reason)
        result[row['task_cid']]=dict(row)
    need(set(result)==set(scope),reason)
    return result


def _catalogue_race_observed_dependencies_quiet(native,profile):
    """Observe selected-parser dependency statuses without revalidating its DAG.

    Legacy proofs have no boundary observation. The explicit envelope mode must
    retain the established generic diagnostic scope and deny every nonquiet or
    unknown observed dependency. No child status satisfies a primary task.
    """
    if 'population_schema'not in native and 'dependency_boundary'not in native:
        return
    h=health_owner();graph=native.get('dependency_boundary')
    need(native.get('population_schema')=='legal-ir-native-catalogue-boundary/v1'
         and type(graph)is dict and graph.get('schema')=='native-observational-dependency-graph/v1'
         and all(graph.get(key)is False for key in ('prerequisite_body_semantics_qualified',
             'completion_authority','dispatch_authority','observed_task_completion_established')),
         'catalogue_race_observed_dependency_scope_or_authority')
    reference=graph.get('population_envelope_ref')
    h._keys(reference,{'path','bytes','sha256'},'catalogue_race_observed_population_descriptor')
    need(h._text(reference['path'])and h._integer(reference['bytes'],1,4*1024*1024)
         and reference['sha256']==profile['catalogue']['population_sha256'],
         'catalogue_race_observed_population_binding')
    rows=graph.get('observed_tasks');scope=profile['task_scope']
    need(type(rows)is list and h._integer(graph.get('observed_task_count'),0,1000-len(scope))
         and len(rows)==graph['observed_task_count']
         and graph.get('node_count')==len(scope)+len(rows)
         and type(graph.get('node_count'))is int,
         'catalogue_race_observed_dependency_population')
    seen=set()
    for row in rows:
        need(type(row)is dict and h._text(row.get('task_cid'))and row['task_cid']not in scope
             and row['task_cid']not in seen and row.get('status')in QUIET_STATUSES
             and h._integer(row.get('revision'),1),
             'catalogue_race_observed_dependency_not_quiet')
        seen.add(row['task_cid'])


def closed_catalogue_race_rejection(proof_ref, profile, identity):
    """Recheck the unmodified original proof, accepting only its exact rejection.

    The original verifier still refuses this proof. Its exception is evidence
    about a closed negative, not an alternate successful health-verifier mode.
    Recorded clocks are used only for historical original-cycle verification.
    A caller that polls again must independently require fresh current native
    data and the unchanged current Main before writing a negative witness.
    """
    h=health_owner();proof=h._document(proof_ref)
    need(proof.get('kind')=='genuine_scoped_idle','catalogue_race_not_idle')
    native_value=h._document(proof['native_diagnostic_ref'])
    native=h._native(native_value,profile,now=proof['finished_at'])
    try:
        h._cycle(proof_ref,profile,identity,native,now=proof['finished_at'],
                 monotonic_ns=proof['finished_monotonic_ns'])
    except h.ExecutorHealthError as error:
        reason=str(error)
        need(reason in CATALOGUE_RACE_REJECTIONS,'catalogue_race_not_exact_verifier_rejection')
    else:
        raise ExecutorHold('catalogue_race_original_proof_was_valid')
    # Everything preceding the original cursor/tasks predicates was checked by
    # h._cycle: original owner/request, exact source and import origins, parent
    # birth/ancestry, original waits, released storage/native and stopped tree.
    # Check every remaining idle predicate explicitly without rewriting data.
    result=h._document(proof['result_ref']);request=h._document(proof['request_ref'])
    terminal=h._document(proof['terminal_ref']);scope=profile['task_scope']
    need(result.get('schema')=='native-embedded-daemon-observation@1'
         and result.get('task_scope')==scope
         and type(result.get('ready_task_count'))is int and result['ready_task_count']==0
         and result.get('selection_idle_reason')=='no_scoped_ready_tasks'
         and type(result.get('write_count'))is int and result['write_count']==0
         and result.get('unchanged')is True and all(result.get(k)is False for k in
            ('provider_dispatched','attempt_consumed','completion_authority')),
         'catalogue_race_outcome_not_quiet_readonly')
    need(terminal.get('absolute_resource_operation_deadline_closed')is True
         and terminal.get('task_completion_authority_granted_by_resource_owner')is False
         and terminal.get('database_preimage_before_grants')==request.get('database_preimage')
         ==terminal.get('database_preimage_observed')==terminal.get('database_postimage'),
         'catalogue_race_original_read_or_deadline_not_closed')
    _catalogue_race_observed_dependencies_quiet(native_value,profile)
    before=_catalogue_race_task_rows(result['catalogue_binding']['task_statuses'],scope,
        'catalogue_race_embedded_task_rows')
    selected=[{k:row[k]for k in ('task_cid','status','revision')}for row in native_value['tasks']
        if row.get('task_cid')in scope]
    after=_catalogue_race_task_rows(selected,scope,'catalogue_race_closing_task_rows')
    previous=result['catalogue_binding']['event_cursor'];current=native_value['event_cursor']
    # A same-cursor state mutation has no observed external event boundary.
    # Recognizing tasks_changed never grants authority to normalize it.
    need(h._integer(previous)and h._integer(current)and previous<current,
         'catalogue_race_no_forward_external_cursor')
    changed=[]
    for cid in scope:
        a,b=before[cid],after[cid]
        need(b['revision']>=a['revision'],'catalogue_race_revision_regressed')
        if a!=b:
            need(b['revision']>a['revision'],'catalogue_race_status_changed_without_revision')
            changed.append({'task_cid':cid,'before':a,'after':b})
    return {'schema':'agent-supervisor/closed-catalogue-race-recognition@1',
        'rejection':reason,'cycle_proof_ref':proof_ref,'from_cursor':previous,'to_cursor':current,
        'task_changes':changed,'native_diagnostic_ref':proof['native_diagnostic_ref'],
        'health_or_task_credit':False,'original_proof_still_rejected':True}


def catalogue_race_deferral(profile_ref, profile, proof_ref, native, kind, process, stop, *, cycle_handle):
    h=health_owner()
    need(kind=='genuine_scoped_idle'and not stop.is_set()and type(process)is subprocess.Popen
         and type(process.returncode)is int and process.returncode==0 and process.poll()==0,
         'catalogue_race_requires_actual_reaped_idle_owner')
    identity=require_main_process(profile_ref,profile)
    h._native(native,profile)  # Current clock, never the historical clock above.
    proof=h._document(proof_ref);root=Path(proof_ref['path']).parent
    fields=h.cycle_start_fields(cycle_handle)
    need(all(proof.get(key)==value for key,value in fields.items()),
         'catalogue_race_actual_original_cycle_capability_changed')
    need(proof_ref['path']==str(root/'cycle-proof.json')
         and root.parent==Path(profile['proof_root'])
         and process.pid==proof['owner_observation']['pid']
         and proof['native_diagnostic_ref']==h.descriptor(root/'native-diagnostic.json')
         and h._document(proof['native_diagnostic_ref'])==native
         and not os.path.lexists(root/'closed.json'), 'catalogue_race_actual_original_cycle_join')
    recognition=closed_catalogue_race_rejection(proof_ref,profile,identity)
    need(not stop.is_set(),'catalogue_race_interrupted')
    return put(root/'catalogue-race-deferred.json',dict(
        schema=CATALOGUE_RACE_NEGATIVE_SCHEMA,recognition=recognition,
        consuming_main_identity=identity,original_wait_returncode=0,sequence=proof['sequence'],
        recorded_at=time.time(),health_sequence_advanced=False,
        health_or_task_completion_authority=False,automatic_retry_of_original_request=False))


def _catalogue_race_authored_population_document(reference):
    """Bound ordinary historical authored DATA, preserving its group-write mode.

    Only the new explicit overlay caller uses this reader, after selecting the
    exact historical population role. Source/control readers stay unchanged.
    The descriptor and file stamp must match one bounded stable observation;
    mutability is never a waiver of the authenticated authored bytes.
    """
    h=health_owner();h._keys(reference,{'path','bytes','sha256'},'catalogue_race_authored_descriptor_shape')
    need(h._integer(reference['bytes'],1,4*1024*1024)
         and type(reference['sha256'])is str and len(reference['sha256'])==64
         and set(reference['sha256'])<=set('0123456789abcdef'),'catalogue_race_authored_descriptor_bounds')
    path=h._path(reference['path'])
    stamp=lambda info:(info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_gid,
        info.st_nlink,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb')as stream:
        first=os.fstat(stream.fileno())
        need(stat.S_ISREG(first.st_mode)and first.st_nlink==1 and first.st_uid==os.geteuid()
             and not first.st_mode&0o002 and first.st_size==reference['bytes'],
             'catalogue_race_authored_DATA_file_unsafe')
        raw=stream.read(reference['bytes']+1);last=os.fstat(stream.fileno())
    need(len(raw)==reference['bytes']and stamp(first)==stamp(last)==stamp(path.lstat())
         and hashlib.sha256(raw).hexdigest()==reference['sha256'],
         'catalogue_race_authored_DATA_changed')
    value=h._decode(raw)
    need(type(value)is dict,'catalogue_race_authored_DATA_not_object')
    return value


def _catalogue_race_population_boundary(reference, review_ref, evidence_ref, prior, profile, oldconfig, config):
    """Bind the reviewed generic DATA overlay without replacing authored tasks.

    The explicitly source-qualified parser validates the graph on every native
    read. This administrative join authenticates its selected DATA and the
    accepted original read from which it was compiled. It adds no satisfied
    dependencies, completion authority, or historical health credit.
    """
    h=health_owner();boundary=h._document(reference);review=h._document(review_ref)
    envelope_ref=h.descriptor(config['catalogue']['population_path']);envelope=h._document(envelope_ref)
    h._keys(envelope,{'schema','authored_population_ref','dependency_boundary_ref'},
        'catalogue_race_population_envelope_shape')
    h._keys(boundary,{'schema','board_namespace','plan_root_cid','repository_tree_id',
        'authored_population_sha256','authored_task_count','authored_goal_count',
        'authored_dependency_count','root_dependencies','observed_tasks','completion_authority'},
        'catalogue_race_graph_boundary_shape')
    authored_ref=envelope['authored_population_ref']
    need(envelope['schema']=='legal-ir-native-catalogue-boundary/v1'
         and envelope['dependency_boundary_ref']==reference
         and boundary['schema']=='legal-ir-native-dependency-boundary/v2'
         and boundary['completion_authority']is False
         and authored_ref['path']==oldconfig['catalogue']['population_path']
         and authored_ref['sha256']==prior['catalogue']['population_sha256']
         and envelope_ref['sha256']==config['catalogue']['population_sha256']
             ==profile['catalogue']['population_sha256']
         and boundary['authored_population_sha256']==authored_ref['sha256'],
         'catalogue_race_population_envelope_binding')
    authored=_catalogue_race_authored_population_document(authored_ref)
    tasks=authored['tasks'];scope=profile['task_scope'];selected={row['task_cid']:row for row in tasks}
    need(len(selected)==len(tasks)==64 and set(selected)==set(scope)==set(prior['task_scope'])
         and boundary['authored_task_count']==64 and type(boundary['authored_task_count'])is int
         and boundary['authored_goal_count']==len(authored['goals'])==34
         and type(boundary['authored_goal_count'])is int
         and boundary['authored_dependency_count']==sum(len(row.get('dependencies',[]))for row in tasks)==115
         and type(boundary['authored_dependency_count'])is int
         and all(boundary[key]==profile['catalogue'][key]for key in
                 ('board_namespace','plan_root_cid','repository_tree_id')),
         'catalogue_race_authored_scope_changed')
    need(review.get('schema')=='legal-ir-proposed-boundary-DATA-independent-result-review/v1'
         and review.get('result')=='accepted_as_exact_PROPOSED_observational_DATA_only'
         and review.get('population_envelope')==envelope_ref
         and review.get('dependency_boundary')==reference
         and review.get('observation_evidence')==evidence_ref
         and type(review.get('observer_source'))is dict
         and all(review['observer_source'].get(key)==catalogue_parser_reference(profile)[key]
                 for key in ('bytes','sha256'))
         and review.get('checks',{}).get('actual_source_and_artifact_joins_match_original_closed_read')is True
         and review['checks'].get('current_status_not_pinned_by_certificate')is True
         and review.get('limits',{}).get('completion_authority')is False
         and review['limits'].get('dispatch_authority')is False
         and review['limits'].get('state')=='proposed',
         'catalogue_race_boundary_independent_DATA_join')
    evidence=h._document(evidence_ref)
    need(evidence.get('schema')=='legal-ir-proposed-boundary-observation-evidence/v1'
         and evidence.get('state')=='proposed'
         and evidence.get('raw_actual_snapshot_ref')==review.get('original_snapshot')
         and evidence.get('original_owner_closeout_ref')==review.get('original_owner_closeout')
         and evidence.get('driver_source_ref')==review.get('driver_source')
         and evidence.get('observer_source_ref')==review.get('observer_source')
         and all(evidence.get(key)is False for key in ('completion_authority','dispatch_authority',
             'publication_or_adoption_performed','status_or_skipped_satisfaction_inferred'))
         and evidence.get('status_observations_come_only_from_raw_snapshot')is True,
         'catalogue_race_boundary_observation_evidence_join')
    original_review_ref=review['prior_independent_closed_read_review']
    original_review=h._document(original_review_ref)
    need(original_review.get('schema')=='legal-ir-original-bounded-closure-read-independent-result-review/v1'
         and original_review.get('result')=='accepted_original_closed_read_and_complete_observed_dependency_graph'
         and original_review.get('snapshot')==evidence['raw_actual_snapshot_ref']
         and original_review.get('original_closeout')==evidence['original_owner_closeout_ref']
         and original_review.get('actual_driver')==evidence['driver_source_ref']
         and original_review.get('checks',{}).get('source_native_process_custody_closed')is True
         and original_review['checks'].get('native_released')is True
         and original_review['checks'].get('storage_status')=='released'
         and original_review.get('acceptance_limits',{}).get('completion_authority')is False
         and original_review['acceptance_limits'].get('dispatch_authority')is False,
         'catalogue_race_original_boundary_read_acceptance_join')
    original=h._document(evidence['original_owner_closeout_ref'])
    need(original.get('schema')=='legal-ir-native-executor-health-original-capture/v1'
         and original.get('returncode')==0 and type(original.get('returncode'))is int
         and original.get('received_signals')==[]and original.get('cleanup_errors')==[]
         and all(original.get(key)is True for key in ('completed','child_reaped',
             'original_CLI_exit_observed','source_READ_closed','source_custody_closed',
             'source_pins_unchanged','original_native_compute_closed'))
         and original.get('native_task_or_service_mutations')is False
         and original.get('plan')==original_review.get('actual_plan')
         and original.get('request')==original_review.get('actual_request'),
         'catalogue_race_original_boundary_read_not_closed')
    snapshot_ref=evidence['raw_actual_snapshot_ref']
    need(type(snapshot_ref['bytes'])is int and 0<snapshot_ref['bytes']<=4*1024*1024,
         'catalogue_race_original_boundary_read_bound')
    snapshot=h._document(snapshot_ref)
    need(snapshot.get('schema')=='legal-ir-atomic-task-dependency-boundary-observation/v2'
         and snapshot.get('original_population')==authored_ref
         and snapshot.get('original_owned_validation_launch')==evidence.get('original_launch')
         and snapshot.get('readonly_physical_connection')is True
         and snapshot.get('transaction')=='BEGIN TRANSACTION; ROLLBACK'
         and type(snapshot.get('native_claim_status_completion_provider_or_evidence_writes'))is int
         and snapshot['native_claim_status_completion_provider_or_evidence_writes']==0
         and snapshot.get('database_and_wal_unchanged')is True
         and snapshot.get('database_before')==snapshot.get('database_after')
         and snapshot.get('completion_authority')is False and snapshot.get('dispatch_authority')is False
         and snapshot.get('skipped_is_not_satisfied')is True
         and snapshot['database_before']['database']['path']==config['catalogue']['database_path'],
         'catalogue_race_original_boundary_snapshot_join')
    closure=snapshot['dependency_closure'];watermark=snapshot['event_watermark']
    need(closure.get('schema')=='legal-ir-atomic-dependency-closure-observation/v1'
         and closure.get('all_outgoing_targets_observed')is True
         and closure.get('all_reads_share_transaction_watermark')==watermark
         and closure.get('seed_task_count')==64
         and h._integer(closure.get('total_task_count'),64,1000)
         and closure.get('maximum_dependency_depth')==16
         and closure.get('maximum_dependency_edges')==10000
         and closure.get('maximum_observed_depth')<=16
         and closure.get('completion_authority')is False and closure.get('dispatch_authority')is False,
         'catalogue_race_original_dependency_closure_incomplete')
    # Runtime graph/schema checks remain owned by the selected parser. These
    # counts and refs join its future native result to this original observation.
    need(len(boundary['observed_tasks'])==closure['noncore_task_count']
         and closure['total_task_count']==64+len(boundary['observed_tasks'])
         and boundary['root_dependencies']==snapshot['current_root_dependency_rows'],
         'catalogue_race_boundary_original_graph_join')
    for ref in (reference,review_ref,evidence_ref,envelope_ref,original_review_ref):
        need(not Path(ref['path']).lstat().st_mode&0o222,'catalogue_race_boundary_not_frozen')
    need(_catalogue_race_authored_population_document(authored_ref)==authored,
         'catalogue_race_authored_population_changed_during_join')
    return dict(boundary=boundary,boundary_ref=reference,envelope_ref=envelope_ref,
        authored_ref=authored_ref,authored=authored,evidence=evidence,snapshot=snapshot,
        snapshot_ref=snapshot_ref,original_review=original_review)


def _catalogue_race_current_progress(bundle, recognition, profile, fresh):
    """Require an accepted same-watermark snapshot for all quiet current rows.

    Later legitimate progress can use a newly accepted observation. A cursor
    after the selected observation has no verified history here and stops;
    historical closing271 is never substituted for a fresh current cursor.
    Old missing revisions below an unchanged starting revision are retained.
    """
    h=health_owner();snapshot=bundle['snapshot'];scope=profile['task_scope']
    need(fresh['event_cursor']==snapshot['event_watermark']>=recognition['to_cursor'],
         'catalogue_race_current_cursor_not_observed')
    closing=h._document(recognition['native_diagnostic_ref'])
    before=_catalogue_race_task_rows([{k:r[k]for k in ('task_cid','status','revision')}
        for r in closing['tasks']if r.get('task_cid')in scope],scope,'catalogue_race_closing_core_rows')
    current=_catalogue_race_task_rows([{k:r[k]for k in ('task_cid','status','revision')}
        for r in fresh['tasks']if r.get('task_cid')in scope],scope,'catalogue_race_fresh_core_rows')
    observed=_catalogue_race_task_rows([{k:r[k]for k in ('task_cid','status','revision')}
        for r in snapshot['original_tasks']],scope,'catalogue_race_observed_core_rows')
    need(current==observed,'catalogue_race_current_rows_not_observed')
    authored={r['task_cid']:r for r in bundle['authored']['tasks']}
    histories=snapshot['task_revision_history_headers']
    need(type(histories)is list and len(histories)<=1000,'catalogue_race_history_bound')
    history={r['task_cid']:r for r in histories}
    need(len(history)==len(histories)and set(scope)<=set(history),'catalogue_race_history_population')
    for row in snapshot['original_tasks']:
        cid=row['task_cid'];old=before[cid];new=current[cid];header=history[cid]
        need(all(type(row[key])is type(authored[cid][key])and row[key]==authored[cid][key]
             for key in ('task_alias','goal_cid','plan_cid','objective_id'))
             and row.get('board_namespace_present')is True
             and row.get('board_namespace')==profile['catalogue']['board_namespace']
             and new['revision']>=old['revision'], 'catalogue_race_current_core_identity_or_revision')
        need(header.get('schema')=='ipfs_accelerate_py/agent-supervisor/task-revision-history-projection@1'
             and header.get('current_task_revision')==new['revision']
             and header.get('current_task_status')==new['status']
             and header.get('current_revision_history_matches')is True
             and header.get('history_has_revisions_above_current')is False,
             'catalogue_race_current_history_join')
        revisions=header['revisions'];indexed={r['revision']:r for r in revisions}
        need(len(indexed)==len(revisions)and len(revisions)<=10000
             and all(h._integer(r['revision'],1)and h._text(r['status'])for r in revisions),
             'catalogue_race_revision_history_invalid')
        # Require only the relevant changed interval. LIR0102's historical r4
        # gap remains honest when its unchanged starting/current revision is r7.
        need(all(number in indexed for number in range(old['revision'],new['revision']+1))
             and indexed[old['revision']]['status']==old['status']
             and indexed[new['revision']]['status']==new['status'],
             'catalogue_race_relevant_revision_history_gap')
    graph=fresh.get('dependency_boundary',{})
    need(fresh.get('population_schema')=='legal-ir-native-catalogue-boundary/v1'
         and graph.get('schema')=='native-observational-dependency-graph/v1'
         and graph.get('authored_population_ref')==bundle['authored_ref']
         and graph.get('dependency_boundary_ref')==bundle['boundary_ref']
         and graph.get('population_envelope_ref')==bundle['envelope_ref']
         and graph.get('authored_dependency_count')==115
         and graph.get('native_outgoing_dependency_count')==115+len(bundle['boundary']['root_dependencies'])
         and graph.get('observed_task_count')==len(bundle['boundary']['observed_tasks'])
         and all(graph.get(key)is False for key in ('prerequisite_body_semantics_qualified',
             'completion_authority','dispatch_authority','observed_task_completion_established')),
         'catalogue_race_current_dependency_graph_join')
    nodes={r['task_cid']:r for r in bundle['boundary']['observed_tasks']}
    rows=graph.get('observed_tasks')
    need(type(rows)is list and len(rows)==len(nodes)and len({r['task_cid']for r in rows})==len(nodes)
         and {r['task_cid']for r in rows}==set(nodes),'catalogue_race_current_dependency_scope')
    observed_noncore=_catalogue_race_task_rows([{k:row[k]for k in ('task_cid','status','revision')}
        for row in snapshot['noncore_dependency_tasks']],list(nodes),'catalogue_race_observed_dependency_rows')
    fresh_noncore=_catalogue_race_task_rows([{k:row[k]for k in ('task_cid','status','revision')}
        for row in rows],list(nodes),'catalogue_race_fresh_dependency_rows')
    need(fresh_noncore==observed_noncore,'catalogue_race_current_dependency_status_not_observed')
    for row in rows:
        node=nodes[row['task_cid']]
        need(row.get('status')in QUIET_STATUSES and h._integer(row.get('revision'),node['minimum_revision'])
             and all(type(row.get(key))is type(node[key])and row[key]==node[key]
                     for key in ('task_alias','goal_cid','plan_cid','objective_id','namespace_binding','observation_role'))
             and row.get('body_binding_profile')==node['body_binding']['profile'],
             'catalogue_race_current_dependency_not_quiet_or_bound')
    return fresh


def _catalogue_race_external_progress(reference, recognition, proof, result, *, boundary_observation):
    """Reconcile only events observed in the accepted original same-txn read.

    No new native snapshot or event schema is invented. The exact historical
    failed-cycle window is selected from the retained full raw event page;
    any status change without its actual independent evidence is refused.
    """
    h=health_owner();evidence=h._document(reference);bundle=boundary_observation
    need(evidence==bundle['evidence'],'catalogue_race_external_observation_crossed')
    snapshot=bundle['snapshot'];page=snapshot['full_event_page'];coverage=snapshot['event_page_coverage']
    low,high=recognition['from_cursor'],snapshot['event_watermark']
    need(type(page)is list and 0<len(page)<=1000 and h._integer(high,low+1)
         and coverage.get('after_global_sequence_exclusive')==low
         and coverage.get('through_snapshot_watermark_inclusive')==high
         and coverage.get('complete_to_snapshot_watermark')is True
         and coverage.get('global_sequence_contiguous')is True
         and coverage.get('row_count')==len(page)
         and [row.get('global_sequence')for row in page]==list(range(low+1,high+1)),
         'catalogue_race_external_fullpage_gap')
    events=[row for row in page if low<row['global_sequence']<=recognition['to_cursor']]
    snap=dict(from_cursor=low,to_cursor=recognition['to_cursor'],events=events)
    events=snap['events'];changes=recognition['task_changes']
    need(type(events)is list and 0<len(events)<=128 and len(events)==2*len(changes)
         and len(changes)<=64 and len(events)==snap['to_cursor']-snap['from_cursor'],
         'catalogue_race_external_event_population')
    need([r.get('global_sequence')for r in events]==list(range(snap['from_cursor']+1,snap['to_cursor']+1)),
         'catalogue_race_external_event_gap')
    from datetime import datetime,timezone
    for row in events:
        need(type(row)is dict and h._text(row.get('recorded_at'))and type(row.get('body'))is dict,
             'catalogue_race_external_event_shape')
        try:
            timestamp=datetime.fromisoformat(row['recorded_at'].replace('Z','+00:00'))
            need(timestamp.tzinfo is not None and timestamp.utcoffset()==timezone.utc.utcoffset(timestamp),
                 'catalogue_race_external_event_timezone')
            epoch=timestamp.timestamp()
        except (TypeError,ValueError,OverflowError) as error:
            raise ExecutorHold('catalogue_race_external_event_clock')from error
        need(h._number(epoch)and proof['started_at']<=epoch<=proof['finished_at']
             and row['body'].get('recorded_at')==row['recorded_at']
             and row['body'].get('event_type')==row.get('event_type'),
             'catalogue_race_external_event_not_during_original_cycle')
    # Each evidence/report descriptor comes from the actual original event,
    # whose independent read acceptance was already joined above.
    evidence=[]
    for change in changes:
        rows=[row for row in events if row.get('task_cid')==change['task_cid']
              and row.get('event_type')=='intent.evidence_recorded']
        need(len(rows)==1,'catalogue_race_actual_completion_evidence_missing')
        payload=rows[0]['body']['body']['body']
        evidence.append(dict(task_cid=change['task_cid'],manifest_ref=payload.get('manifest'),
                             report_ref=payload.get('report')))
    by={row['task_cid']:row for row in evidence}
    for change in changes:
        cid=change['task_cid'];rows=[r for r in events if r.get('task_cid')==cid]
        need(len(rows)==2 and [r.get('event_type')for r in rows]==[
            'intent.evidence_recorded','intent.completion_recorded'], 'catalogue_race_external_event_pair')
        first,last=(r['body']for r in rows);item=first['body'];completion=last['body']
        need(first.get('schema')==last.get('schema')=='ipfs_accelerate_py/agent-supervisor/intent-event@1'
             and h._text(first.get('owner_id'))and first['owner_id']==last.get('owner_id')
             and first.get('subject_id')==item.get('evidence_id')
             and last.get('subject_id')==cid and item.get('task_cid')==completion.get('task_cid')==cid
             and item.get('digest')in completion.get('evidence_digests',[])
             and completion.get('previous_status')==change['before']['status']
             and completion.get('status')==change['after']['status']
             and completion.get('revision')==change['after']['revision']
             and change['after']['revision']==change['before']['revision']+1,
             'catalogue_race_external_completion_join')
        refs=by[cid];h._keys(refs,{'task_cid','manifest_ref','report_ref'},'catalogue_race_completion_refs')
        receipt=completion.get('receipt',{});payload=item.get('body',{})
        need(all(receipt.get(key)is False for key in ('attempt_budget_reset','attempts_rearmed',
            'dispatch_authorized','launcher_invoked','native_admission','new_training_launched',
            'training_authorized','formalized','admitted','gold_or_teacher_admission_granted',
            'Lean_admitted','task_completion_authority')),
            'catalogue_race_external_completion_not_metadata_only')
        need(receipt.get('manifest')==payload.get('manifest')==refs['manifest_ref']
             and receipt.get('report')==payload.get('report')==refs['report_ref'],
             'catalogue_race_external_evidence_descriptor_join')
        h._bound(refs['manifest_ref']);report=h._document(refs['report_ref'])
        need(report.get('manifest')==refs['manifest_ref']and report.get('task_cid')==cid,
             'catalogue_race_independent_completion_report_join')
    return {'event_window':events,'task_changes':changes,'health_or_task_credit':False}



def _catalogue_race_unit_projection(argv,entrypoint):
    argv=list(argv)
    parser_flags=('--catalogue-parser-source','--catalogue-parser-sha256')
    if any(flag in argv for flag in parser_flags):
        need(all(argv.count(flag)==1 and argv.index(flag)+1<len(argv)for flag in parser_flags),
             'catalogue_race_unit_parser_pair')
        for flag in parser_flags:
            index=argv.index(flag);del argv[index:index+2]
    flags=[x for x in ('--known-source-admission-recovery','--known-closed-catalogue-race-recovery')if x in argv]
    need(len(flags)==1 and argv.count(flags[0])==1,'catalogue_race_unit_mode')
    normalized=list(argv);index=normalized.index(flags[0]);need(index+1<len(normalized),'catalogue_race_unit_mode_path')
    normalized[index:index+2]=['--known-source-admission-recovery','<recovery>']
    return source_admission_unit_projection(normalized,entrypoint)


def catalogue_parser_reference(profile):
    """The optional parser is selected by two literal unit argv slots only."""
    h=health_owner();argv=profile['unit']['argv']
    flags=('--catalogue-parser-source','--catalogue-parser-sha256')
    need(all(argv.count(flag)==1 and argv.index(flag)+1<len(argv)for flag in flags),
         'catalogue_parser_explicit_pair_required')
    path=argv[argv.index(flags[0])+1];sha=argv[argv.index(flags[1])+1]
    producer=Path(profile['main_entrypoint']['path']).parent
    need(path==str(producer/'native_catalogue_probe.py')and len(sha)==64
         and set(sha)<=set('0123456789abcdef'),'catalogue_parser_exact_sibling_or_digest')
    selected=[row for row in profile['source_descriptors']if row['path']==path]
    need(len(selected)==1 and selected[0]['sha256']==sha,'catalogue_parser_not_profile_pinned')
    h._bound(selected[0]);return selected[0]


def catalogue_race_source_inventory(reference,profile,parser_ref):
    """Explicit @2 producer commitment binds exactly five declared source files.

    The old four-file producer contract remains unchanged for its old modes.
    This new join grants no SDK/cache/health replacement or source discovery.
    """
    h=health_owner();source=h._document(reference)
    h._keys(source,{'schema','source_root','source_commit','source_tree','source_inventory',
        'source_operation_plan','source_qualification','runtime_activation_or_task_authority',
        'catalogue_parser'},'catalogue_parser_source_commitment_shape')
    producer=Path(profile['main_entrypoint']['path']).parents[3]
    need(source['schema']=='agent-supervisor/scientific-producer-source-commitment@2'
         and source['source_root']==str(producer)and source['catalogue_parser']==parser_ref
         and source['runtime_activation_or_task_authority']is False
         and parser_ref==catalogue_parser_reference(profile)
         and all(type(source[k])is str and len(source[k])==40 and set(source[k])<=set('0123456789abcdef')
                 for k in ('source_commit','source_tree')),'catalogue_parser_source_commitment')
    invref=source['source_inventory']
    need(any(root in Path(invref['path']).parents for root in ORIGINAL_NAMED_ROOTS),
         'catalogue_parser_inventory_not_durable')
    inventory=h._document(invref);rows=inventory.get('files',[])
    expected={str(producer/'ipfs_accelerate_py/__init__.py'),
        str(producer/'ipfs_accelerate_py/agent_supervisor/__init__.py'),
        str(producer/'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py'),
        profile['main_entrypoint']['path'],parser_ref['path']}
    indexed={r['path']:r for r in rows}
    selected={p['path']:p for p in profile['source_descriptors']if producer in Path(p['path']).parents}
    need(inventory.get('schema')=='legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(k)==source[k]for k in ('source_root','source_commit','source_tree'))
         and inventory.get('file_count')==len(rows)==len(indexed)==len(selected)==5
         and set(indexed)==set(selected)==expected,'catalogue_parser_exact_five_source_population')
    for path,pin in selected.items():
        need(all(indexed[path].get(k)==pin[k]for k in ('path','bytes','sha256'))
             and indexed[path].get('mode')==stat.S_IMODE(Path(path).lstat().st_mode)
             and not Path(path).lstat().st_mode&0o222,'catalogue_parser_inventory_bytes_or_modes_changed')
        h._bound(pin)
    for key in ('source_inventory','source_operation_plan','source_qualification'):
        h._bound(source[key])
    return inventory


class _ExplicitCatalogueParser:
    __slots__=('descriptor','module','observer')


def load_explicit_catalogue_parser(reference):
    """Compile authenticated raw bytes at the real filename, bypassing pyc.

    This is an opaque metadata parser import. It makes no normal package or
    sys.path claim and never accepts a preexisting module as source authority.
    Callers must first establish the explicit profile/five-file source join.
    """
    import importlib.machinery
    import importlib.util
    import types
    h=health_owner();raw=h._bound(reference);path=Path(reference['path'])
    need(not path.lstat().st_mode&0o222,'catalogue_parser_source_must_be_readonly')
    class AuthenticatedRawLoader(importlib.machinery.SourceFileLoader):
        def get_code(self,fullname):
            offered=h._bound(reference)
            need(offered==raw,'catalogue_parser_source_changed_before_compile')
            return compile(offered,str(path),'exec',dont_inherit=True)
    name='_ipfs_scientific_catalogue_parser_'+reference['sha256']
    need(name not in sys.modules,'catalogue_parser_opaque_name_already_present')
    loader=AuthenticatedRawLoader(name,str(path))
    spec=importlib.util.spec_from_file_location(name,str(path),loader=loader)
    need(spec is not None and spec.origin==str(path),'catalogue_parser_real_source_spec_required')
    module=importlib.util.module_from_spec(spec);module.__package__='';sys.modules[name]=module
    try:
        loader.exec_module(module)
        need(h._bound(reference)==raw,'catalogue_parser_source_changed_during_execution')
        observer=module.__dict__.get('observe_native_catalogue')
        need(module.__file__==str(path)and module.__package__==''and module.__spec__.origin==str(path)
             and type(observer)is types.FunctionType and observer.__module__==name
             and observer.__code__.co_filename==str(path),'catalogue_parser_callable_origin_changed')
        result=_ExplicitCatalogueParser();result.descriptor=dict(reference)
        result.module=module;result.observer=observer;return result
    except BaseException:
        if sys.modules.get(name)is module:del sys.modules[name]
        raise


def catalogue_race_native_owner_qualification(reference,owner):
    """Port of the shared exact Owner128 reader; receipts are DATA, never code.

    The reviewed contract is embedded as source DATA to avoid importing a TMP
    helper. Original execution paths remain exact; deployment joins use bytes
    and SHA by explicit source role. No live or task authority is inferred.
    """
    import xml.etree.ElementTree as ET
    h=health_owner();document=h._document;exact=h._bound;descriptor=h.descriptor;raw=h._raw
    spec={"future_owner128_acceptance":reference}
    OWNER128_ACCEPTANCE_KEYS=frozenset(['schema', 'accepted', 'original_root_closeout', 'plan', 'request', 'scope', 'observation', 'owner_source', 'owning_test_source', 'cases', 'natural_phases', 'selector_contract_mode', 'raw_subjects', 'all_raw_subjects_compiled_and_executed', 'original_source_native_process_storage_closed', 'runtime_activation_or_task_authority'])
    OWNER128_METADATA={'plan': {'path': '/tmp/legal-ir-native-owner128-controls-20261107-v01/plan-v01.json', 'bytes': 3729, 'sha256': '652d987ab1fc633d6c6f2e05f98d2575d83663208ef20b78930db004403bbdca'}, 'request': {'path': '/tmp/legal-ir-native-owner128-controls-20261107-v01/request-v01.json', 'bytes': 31865, 'sha256': 'e80b5d432d5f4e70404f2c4c0af0049ae488ac88f11d3a4a3688863b190c132b'}, 'scope': {'path': '/tmp/legal-ir-native-owner128-controls-20261107-v01/scope-v01.json', 'bytes': 154253, 'sha256': 'a1c6d4b4178e0ba600f6774e792fa4961cd991f5efa8d82040f8843d5d597d38'}}
    CURRENT_SCHEDULER_SHA='f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
    contract={'acceptance_keys': ['schema', 'accepted', 'original_root_closeout', 'plan', 'request', 'scope', 'observation', 'owner_source', 'owning_test_source', 'cases', 'natural_phases', 'selector_contract_mode', 'raw_subjects', 'all_raw_subjects_compiled_and_executed', 'original_source_native_process_storage_closed', 'runtime_activation_or_task_authority'], 'acceptance_schema': 'legal-ir-native-owner128-source-acceptance/v1', 'actual_acceptance': None, 'authority_supplied': False, 'binding_rules': ['original_root_closeout is exactly the original0bb Root artifact supplied with explicit expected SHA256, not a synthetic summary.', 'plan652d987a/requeste80b5d43/scopea1c6d4b4 authenticate the precise original17 Owner128 request and worker93656 contract.', "observation is the original child's focused-qualification-result.json joined through original terminal artifact inventory and durability.", 'Require actual source/native/process/storage closure, child exit0/reaped, original released reservation and unchanged source/installed inputs before acceptance.', 'Require observation selector_contract_mode whole_owner_file_128, exact collected128 IDs in order, exact natural384 passed phases, JUnit128/0failure/0error/0skip and all four authenticated raw compilations/executions.', 'Preserve exact32978-byte existing test prefix and qualified old75 ordered IDs; all53 additional cases must be observed.', 'owner_source and owning_test_source are frozen actual execution paths. Future Git projection may compare bytes/SHA by relative role; it must not require a not-yet-existing Git publication before this source qualification.', 'Native owner4 Git materialization and source-operation-plan/commitment proofs remain a separate later qualification.', 'Artifact checks, full OWNED durability and no surviving original process/group remain required; old175 source results are not pooled.', 'Source-only acceptance supplies no live restart, native execution route, legal meaning, numerical, graph, Lean or task authority.'], 'descriptor_keys': ['original_root_closeout', 'plan', 'request', 'scope', 'observation', 'owner_source', 'owning_test_source'], 'fixed_values': {'accepted': True, 'all_raw_subjects_compiled_and_executed': True, 'cases': 128, 'natural_phases': 384, 'original_source_native_process_storage_closed': True, 'runtime_activation_or_task_authority': False, 'schema': 'legal-ir-native-owner128-source-acceptance/v1', 'selector_contract_mode': 'whole_owner_file_128'}, 'raw_subject_hashes': {'driver': '2e9ba7653bbc0939bb41f1fbbdd14f5f0e9a51fd78577dbbb5aa044fc14e3ea6', 'guard': '141b7da6a57bd65d03c673c5e1041d8e26bce0ed7c02803a3d11f829925da95e', 'owner': '82b0fd08a2e343350578269ef9d90ec5f6537830434bf5c0cd98135933f0ca13', 'owning_test': '2bdcfad397a96d150bf958570042123dbbeebe2d8087fdb06aed50d8804fc67c'}, 'raw_subject_keys': ['owner', 'owning_test', 'driver', 'guard'], 'schema': 'legal-ir-owner128-acceptance-source-contract/v1', 'source_ref_schema': {'bytes_type': 'exact positive int', 'keys': ['path', 'bytes', 'sha256'], 'path': 'ordinary absolute bounded file', 'sha256': '64 lowercase hex'}, 'strict_types': {'accepted': 'bool is True', 'all_raw_subjects_compiled_and_executed': 'bool is True', 'cases': 'exact int', 'natural_phases': 'exact int', 'original_source_native_process_storage_closed': 'bool is True', 'runtime_activation_or_task_authority': 'bool is False'}}
    accepted=document(spec['future_owner128_acceptance'])
    need(set(accepted)==OWNER128_ACCEPTANCE_KEYS
         and accepted['schema']==contract['acceptance_schema']
         and accepted['accepted'] is True and type(accepted['cases']) is int and accepted['cases']==128
         and type(accepted['natural_phases']) is int and accepted['natural_phases']==384
         and accepted['selector_contract_mode']=='whole_owner_file_128'
         and accepted['all_raw_subjects_compiled_and_executed'] is True
         and accepted['original_source_native_process_storage_closed'] is True
         and accepted['runtime_activation_or_task_authority'] is False
         and owner['source_qualification']==spec['future_owner128_acceptance'],
         'actual exact Owner128 whole-file qualification required; no old175 pooling')
    for key in contract['descriptor_keys']:exact(accepted[key])
    need(all(accepted[key]==OWNER128_METADATA[key] for key in ('plan','request','scope')),
         'Owner128 original request/plan/scope generation differs')
    plan=document(accepted['plan']);request=document(accepted['request']);scope=document(accepted['scope'])
    observation=document(accepted['observation']);root=document(accepted['original_root_closeout'])
    need(root['plan']==accepted['plan'] and root['request']==accepted['request']
         and plan['request']==accepted['request'] and plan['scope']==accepted['scope']
         and root['returncode']==0 and root.get('failure') is None
         and root['cleanup_errors']==root['received_signals']==[]
         and all(root[key] is True for key in ('completed','child_reaped','source_READ_closed',
             'source_custody_closed','original_native_compute_closed','original_CLI_exit_observed',
             'source_pins_unchanged','installed_tools_unchanged')),
         'actual original Owner128 closed root and exact source/native/process chain required')
    need(len(request)==17 and request['schema']=='owned-validation-request/v1'
         and request['declared_external_writable_paths']==[]
         and request['expected_scheduler_policy_sha256']==CURRENT_SCHEDULER_SHA,
         'original Owner128 public17/current-policy request differs')
    selected=scope['selected_checks'];selection=scope['selector_contract']
    need(scope['expected_cases']==128 and scope['expected_phases']==384
         and type(selected) is list and len(selected)==len(set(selected))==128
         and selection['mode']=='whole_owner_file_128' and selection['expanded_checks']==selected
         and observation['schema']=='scientific-executor-focused-observation/v1'
         and observation['passed'] is True and observation['case_count']==128 and observation['phase_count']==384
         and observation['selected_checks']==selected
         and observation['pytest_selected_checks']==selection['pytest_selectors']
         and observation['selector_contract_mode']=='whole_owner_file_128'
         and observation['error'] is None and observation['guard_denials']==[]
         and observation['JUnit']=={'tests':128,'failures':0,'errors':0,'skipped':0}
         and [(row['nodeid'],row['when'],row['outcome']) for row in observation['phases']]==[
             (node,when,'passed') for node in selected for when in ('setup','call','teardown')]
         and observation['whole_owner_file_contract']=={'pytest_target':selection['pytest_selectors'][0],
             'function_case_counts':selection['function_case_counts'],'exact_expanded_collection_observed':True,
             'expected_cases':128,'expected_natural_phases':384},
         'actual exact ordered Owner128 collection and natural384 phases required')
    subjects=accepted['raw_subjects']
    need(type(subjects) is dict and set(subjects)==set(contract['raw_subject_keys'])
         and subjects['owner']==accepted['owner_source']
         and subjects['owning_test']==accepted['owning_test_source']
         and set(scope['required_raw_subjects'])=={pin['path'] for pin in subjects.values()},
         'four closed actual execution-role source descriptors required')
    for role,pin in subjects.items():
        exact(pin)
        need(pin['bytes']>0 and pin['sha256']==contract['raw_subject_hashes'][role]
             and scope['sources'].get(pin['path'])==pin
             and request['source_sha256'].get(pin['path'])==pin['sha256']
             and observation['raw_compiled_sources'].get(pin['path'])==pin['sha256']
             and pin['path'] in observation['executed_raw_paths'],
             'Owner128 actual raw compile/execution role differs: '+role)
    # Projection paths legitimately differ. Body/hash equality by named role is
    # the source qualification join; the inventory above owns new physical paths.
    need(all(owner['owner_source'][key]==subjects['owner'][key] for key in ('bytes','sha256'))
         and all(owner['native_backend_source'][key]==subjects['driver'][key] for key in ('bytes','sha256')),
         'future OW owner or retained H backend differs from actual qualified bodies')
    terminal=root['owner_terminal'];reservation=terminal['reservation'];record=reservation['record']
    need(reservation['status']==record['status']=='released' and record['artifacts_durable_asserted'] is True
         and record['validation_request_sha256']==accepted['request']['sha256']
         and record['attempt_exceeded_reservation'] is False and record['external_charges']=={}
         and record['final_external_charged_bytes']==0 and reservation['resource_lease']['released'] is True
         and reservation['process_observation']['live_processes']==0
         and reservation['process_observation']['tracking_error']==''
         and reservation['process_observation']['out_of_group_live_identities']==[],
         'actual original Owner128 released promise and no surviving sampled group required')
    attempt=Path(request['attempt_directory']);observation_path=attempt/'exports/focused-qualification-result.json'
    owned=json.loads(raw(attempt/'logs/result.json'));execution=json.loads(raw(attempt/'logs/execution.json'))
    need(descriptor(observation_path)==accepted['observation']
         and terminal['result']==str(attempt/'logs/result.json')
         and document(root['stdout'])==terminal
         and owned['schema']==execution['schema']=='owned-validation-result/v1'
         and owned['completed'] is execution['completed'] is True
         and owned['returncode']==execution['returncode']==0
         and owned['request_sha256']==execution['request_sha256']==accepted['request']['sha256']
         and owned['executed_checks']==execution['executed_checks']==request['selected_checks'],
         'Owner128 observation and successful owned result must join the exact original terminal/request')
    durability=json.loads(raw(attempt/'logs/validation-durability.json'))
    need(durability['schema']=='owned-validation-durability/v1'
         and durability['request_sha256']==accepted['request']['sha256']
         and durability['reservation_id']==record['reservation_id'],
         'Owner128 original OWNED concrete durability join required')
    observed_files={str(path.relative_to(attempt)) for path in attempt.rglob('*') if path.is_file()}
    need(observed_files=={row['path'] for row in durability['files']}|{'logs/validation-durability.json'},
         'Owner128 complete ordinary durable file population differs')
    for row in durability['files']:
        path=attempt/row['path'];info=path.lstat()
        need(descriptor(path)=={key:dict(row,path=str(path))[key] for key in ('path','bytes','sha256')}
             and (info.st_dev,info.st_ino)==(row['device'],row['inode']),
             'Owner128 recorded ordinary file bytes or inode differs')
    need(sum(row['bytes'] for row in durability['files'])+len(raw(attempt/'logs/validation-durability.json'))
             ==record['final_attempt_bytes']==record['final_total_charged_bytes']
         and 0<record['final_attempt_bytes']<=record['storage_bytes'],
         'Owner128 released record must bind the complete original durable ordinary byte count')
    junit=ET.fromstring(raw(attempt/'exports/test-results.xml'))
    need({key:sum(int(suite.get(key,'0')) for suite in junit.findall('.//testsuite'))
             for key in ('tests','failures','errors','skipped')}==observation['JUnit'],
         'original Owner128 JUnit must contain128 actual cases with zero failures/errors/skips')
    return accepted


def catalogue_race_native_owner_inventory(reference,prior,profile):
    """Allow only the qualified explicit 4662-to-f07 native-owner transition.

    The prior H owner, H driver/cache and every historical source pin remain
    selected. New owner code has its own four-file immutable source lineage;
    this does not turn a producer inventory into native-owner authority.
    """
    h=health_owner();value=h._document(reference)
    h._keys(value,{'schema','source_root','source_commit','source_tree','source_inventory',
        'source_operation_plan','source_qualification','owner_source','prior_owner_source',
        'native_backend_source','scheduler_policy_transition','runtime_activation_or_task_authority'},
        'catalogue_race_owner_commitment_shape')
    transition={'prior_sha256':'4662bd8fe6caf6f2c048e3d7cff7abb4d80e5c875b21e02e204b1b89371b8286',
        'target_sha256':'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'}
    need(value['schema']=='agent-supervisor/scientific-native-owner-policy-source-commitment@1'
         and value['runtime_activation_or_task_authority']is False
         and value['scheduler_policy_transition']==transition
         and value['prior_owner_source']==prior['owner_source']
         and prior['owner_source']['bytes']==72407
         and prior['owner_source']['sha256']=='e243cdf7b22a63f55e3c55101a2a7aeeee31492a98c8dbfe9f05b4d43b7fdc06'
         and value['owner_source']==profile['owner_source']and profile['owner_source']!=prior['owner_source']
         and value['native_backend_source']==profile['driver_source']==prior['driver_source']
         and profile['driver_source']['bytes']==28162
         and profile['driver_source']['sha256']=='2e9ba7653bbc0939bb41f1fbbdd14f5f0e9a51fd78577dbbb5aa044fc14e3ea6'
         and profile['health_module']==prior['health_module'],
         'catalogue_race_owner_policy_or_backend_transition_changed')
    owner_root=Path(value['source_root']);producer=Path(profile['main_entrypoint']['path']).parents[3]
    native_root=Path(prior['owner_source']['path']).parents[3]
    need(profile['owner_source']['path']==str(owner_root/'ipfs_accelerate_py/agent_supervisor/rescue/original_embedded_observation_owner.py')
         and owner_root==Path(profile['owner_source']['path']).parents[3]
         and all(owner_root!=root and owner_root not in root.parents and root not in owner_root.parents
                 for root in (producer,native_root))
         and all(type(value[k])is str and len(value[k])==40 and set(value[k])<=set('0123456789abcdef')
                 for k in ('source_commit','source_tree')),'catalogue_race_owner_root_or_generation_changed')
    need(any(root in Path(value['source_inventory']['path']).parents for root in ORIGINAL_NAMED_ROOTS),
         'catalogue_race_owner_inventory_not_durable')
    inventory=h._document(value['source_inventory']);rows=inventory.get('files',[])
    expected={str(owner_root/'ipfs_accelerate_py/__init__.py'),
        str(owner_root/'ipfs_accelerate_py/agent_supervisor/__init__.py'),
        str(owner_root/'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py'),profile['owner_source']['path']}
    indexed={r['path']:r for r in rows}
    selected={p['path']:p for p in profile['source_descriptors']if owner_root in Path(p['path']).parents}
    need(inventory.get('schema')=='legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(k)==value[k]for k in ('source_root','source_commit','source_tree'))
         and inventory.get('file_count')==len(rows)==len(indexed)==len(selected)==4
         and set(indexed)==set(selected)==expected,'catalogue_race_owner_exact_four_source_roles')
    for path,pin in selected.items():
        need(all(indexed[path].get(k)==pin[k]for k in ('path','bytes','sha256'))
             and indexed[path].get('mode')==stat.S_IMODE(Path(path).lstat().st_mode)
             and not Path(path).lstat().st_mode&0o222,'catalogue_race_owner_inventory_changed')
        h._bound(pin)
    for key in ('source_inventory','source_operation_plan','source_qualification'):
        h._bound(value[key])
    catalogue_race_native_owner_qualification(value['source_qualification'],value)
    return value,selected


def _catalogue_race_template_transition(reference,owner_ref,owner,owner_pins,prior,profile,oldconfig,config):
    """Compare unissued template DATA; never normalize historical evidence."""
    h=health_owner();certificate=h._document(reference)
    h._keys(certificate,{'schema','native_owner_source_commitment','prior_template_ref','template_ref',
        'scheduler_policy_transition','independent_review_ref','runtime_activation_or_task_authority'},
        'catalogue_race_policy_transition_shape')
    need(certificate['schema']=='agent-supervisor/scientific-native-owner-policy-transition@1'
         and certificate['runtime_activation_or_task_authority']is False
         and certificate['native_owner_source_commitment']==owner_ref
         and certificate['scheduler_policy_transition']==owner['scheduler_policy_transition']
         and certificate['prior_template_ref']==oldconfig['idle_template_ref']
         and certificate['template_ref']==config['idle_template_ref'],
         'catalogue_race_policy_transition_certificate_join')
    oldtemplate=h._document(certificate['prior_template_ref']);newtemplate=h._document(certificate['template_ref'])
    validate_template(oldtemplate,prior,'genuine_scoped_idle');validate_template(newtemplate,profile,'genuine_scoped_idle')
    oldseed=Path(oldtemplate['seed_root']);newseed=Path(newtemplate['seed_root'])
    need(newseed!=oldseed and newseed not in oldseed.parents and oldseed not in newseed.parents,
         'catalogue_race_new_template_not_distinct_same_scope')
    restored=relocate(newtemplate,newseed,oldseed)
    for key in ('request_ref','plan_ref'):restored[key]=oldtemplate[key]
    need(restored==oldtemplate,'catalogue_race_template_shape_or_links_changed')
    oldplan=h._document(oldtemplate['plan_ref']);newplan=h._document(newtemplate['plan_ref'])
    oldrequest=h._document(oldtemplate['request_ref']);newrequest=h._document(newtemplate['request_ref'])
    for request,template,selected in ((oldrequest,oldtemplate,prior),(newrequest,newtemplate,profile)):
        need(request.get('argv')==[selected['unit']['argv'][0],'-B',selected['driver_source']['path'],
            '--plan',template['plan_ref']['path'],'--plan-sha256',template['plan_ref']['sha256']],
            'catalogue_race_template_original_driver_argv_or_plan_digest_changed')
    transition=owner['scheduler_policy_transition']
    need(oldrequest.get('expected_scheduler_policy_sha256')==transition['prior_sha256']
         and newrequest.get('expected_scheduler_policy_sha256')==transition['target_sha256']
         and all(type(request.get(k))is int and request[k]==v for request in (oldrequest,newrequest)
                 for k,v in dict(cpu_slots=1,memory_mb=1536,child_process_slots=8,
                     storage_bytes=384*1024**2,timeout_seconds=300).items())
         and oldrequest['source_roots']==newrequest['source_roots']==[
             '/home/barberb/lift_coding/external/ipfs_datasets','/home/barberb/lift_coding/external/ipfs_accelerate']
         and oldrequest['resource_ledger']==newrequest['resource_ledger']==str(ORIGINAL_RESOURCE_LEDGER)
         and oldrequest['scheduler_state_path']==newrequest['scheduler_state_path'],
         'catalogue_race_original_native_envelope_or_policy_changed')
    # A future prepare_cycle refreshes these two explicit DB descriptor roles.
    # This projection is of unissued templates, never of the failed162 proof.
    for request,plan in ((oldrequest,oldplan),(newrequest,newplan)):
        db=request.get('database_preimage',{})
        need(db.get('path')==oldconfig['catalogue']['database_path']==config['catalogue']['database_path']
             and h._integer(db.get('bytes'),1,128*1024**2)
             and type(db.get('sha256'))is str and len(db['sha256'])==64
             and set(db['sha256'])<=set('0123456789abcdef')
             and plan['native_metadata_cli']['observation_binding']['database_preimage']==db,
             'catalogue_race_template_database_role_changed')
    projected_plan=relocate(newplan,newseed,oldseed)
    projected_plan['native_metadata_cli']['observation_binding']['database_preimage']=oldrequest['database_preimage']
    need(projected_plan==oldplan,'catalogue_race_template_native_backend_or_task_scope_changed')
    projected_request=relocate(newrequest,newseed,oldseed)
    projected_request['database_preimage']=oldrequest['database_preimage']
    projected_request['expected_scheduler_policy_sha256']=transition['prior_sha256']
    projected_request['argv'][projected_request['argv'].index('--plan-sha256')+1]=oldtemplate['plan_ref']['sha256']
    oldpins=dict(oldrequest['source_sha256']);newpins=dict(projected_request['source_sha256'])
    # Only the selected plan's physical descriptor and explicit OW source pins
    # differ. Original H and canonical R source pins cannot be removed/changed.
    need(oldpins.pop(oldtemplate['plan_ref']['path'],None)==oldtemplate['plan_ref']['sha256']
         and newpins.pop(oldtemplate['plan_ref']['path'],None)==newtemplate['plan_ref']['sha256'],
         'catalogue_race_template_plan_descriptor_join')
    for path,pin in owner_pins.items():
        if path in newpins:
            need(newpins[path]==pin['sha256']and path not in oldpins,
                 'catalogue_race_template_owner_source_changed')
            newpins.pop(path)
    need(all(newrequest['source_sha256'].get(path)==pin['sha256']for path,pin in owner_pins.items())
         and newpins==oldpins,'catalogue_race_template_original_source_pin_changed')
    projected_request['source_sha256']=oldrequest['source_sha256']
    need(projected_request==oldrequest,'catalogue_race_template_request_authority_changed')
    review=h._document(certificate['independent_review_ref'])
    need(review.get('schema')=='scientific-native-owner-policy-transition-independent-review/v1'
         and review.get('accepted_explicit_owner_policy_transition')is True
         and review.get('native_owner_source_commitment')==owner_ref
         and review.get('prior_template_ref')==certificate['prior_template_ref']
         and review.get('template_ref')==certificate['template_ref']
         and review.get('scheduler_policy_transition')==transition
         and review.get('runtime_or_task_completion_authority')is False,
         'catalogue_race_policy_transition_independent_review_join')
    for ref in (reference,certificate['independent_review_ref'],certificate['template_ref']):
        need(not Path(ref['path']).lstat().st_mode&0o222,'catalogue_race_policy_transition_not_frozen')
    return certificate


def _catalogue_race_scheduler_transition(request,snapshot_ref,transition):
    """Check historical4662 and currentf07 separately; preserve original proof."""
    h=health_owner();archived=h._document(snapshot_ref);path=Path(request['scheduler_state_path'])
    need(request['expected_scheduler_policy_sha256']==transition['prior_sha256'],
         'catalogue_race_original_policy_not_prior')
    before=path.lstat();need(0<before.st_size<=h.MAX_JSON,'catalogue_race_scheduler_data_bound')
    reference=h.database_descriptor(path)
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb')as stream:
        first=os.fstat(stream.fileno());raw=stream.read(h.MAX_JSON+1);last=os.fstat(stream.fileno())
    need(stamp(before)==stamp(first)==stamp(last)==stamp(path.lstat())
         and len(raw)==reference['bytes']<=h.MAX_JSON
         and hashlib.sha256(raw).hexdigest()==reference['sha256']
         and h.database_descriptor(path)==reference,'catalogue_race_scheduler_data_changed')
    family={request['request_id'],request['request_id']+':root',request['request_id']+':phase'}
    for state,expected in ((archived,transition['prior_sha256']),(h._decode(raw),transition['target_sha256'])):
        need(type(state.get('config'))is dict,'catalogue_race_scheduler_config_missing')
        digest=hashlib.sha256(json.dumps(state['config'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        need(digest==expected,'catalogue_race_scheduler_not_exact_qualified_policy')
        for field in ('leases','waiters'):
            rows=state.get(field)
            need(type(rows)is dict and all(type(row)is dict for row in rows.values())
                 and not any(row.get('request_id')in family for row in rows.values()),
                 'catalogue_race_original_request_still_active_or_unknown')


def _validate_closed_catalogue_race_recovery(value,profile_ref,profile,config):
    h=health_owner();h._keys(value,CATALOGUE_RACE_KEYS,'catalogue_race_recovery_shape')
    need(value['schema']==CATALOGUE_RACE_SCHEMA
         and value['authorization']=='resume_idle_after_proved_closed_catalogue_observation_race'
         and value['profile_ref']==profile_ref and h._profile(profile_ref)==profile,
         'catalogue_race_recovery_authority')
    need(not os.path.lexists(Path(profile['proof_root'])/'catalogue-race-reconciliation-v01.json'),
         'catalogue_race_authorization_already_consumed')
    need(h._number(value['recorded_at'])and 0<=time.time()-value['recorded_at']<=profile['max_age_seconds']
         and value['manager_before'].get('Id')==profile['unit']['name']
         and value['manager_before'].get('MainPID')=='0'
         and value['manager_before'].get('ActiveState')in ('failed','inactive'),
         'catalogue_race_admin_clock_or_manager')
    prior=h._document(value['prior_profile_snapshot_ref']);oldconfig=h._document(value['prior_config_snapshot_ref'])
    need(_same_payload(value['prior_profile_ref'],value['prior_profile_snapshot_ref'])
         and profile['proof_root']==prior['proof_root']and profile['receipt_path']!=prior['receipt_path']
         and Path(profile['proof_root'])in Path(profile['receipt_path']).parents
         and not os.path.lexists(profile['receipt_path']),'catalogue_race_same_root_new_absent_health_epoch')
    for key in ('task_scope','max_age_seconds','max_cycle_seconds','cycle_contract'):
        need(prior[key]==profile[key],'catalogue_race_scope_changed')
    need(all(prior[k]==profile[k]for k in ('health_module','driver_source'))
         and all(p in profile['source_descriptors']for p in prior['source_descriptors']),
         'catalogue_race_original_backend_or_source_removed')
    owner,owner_pins=catalogue_race_native_owner_inventory(value['native_owner_source_commitment_ref'],prior,profile)
    need(all(prior['unit'][k]==profile['unit'][k]for k in ('name','interpreter','working_directory','environment'))
         and _catalogue_race_unit_projection(prior['unit']['argv'],prior['main_entrypoint']['path'])
             ==_catalogue_race_unit_projection(profile['unit']['argv'],profile['main_entrypoint']['path']),
         'catalogue_race_unit_authority_changed')
    oldcat=dict(prior['catalogue']);newcat=dict(profile['catalogue'])
    oldcat.pop('population_sha256');newcat.pop('population_sha256')
    need(oldcat==newcat,'catalogue_race_catalogue_identity_changed')
    oldcat=dict(oldconfig['catalogue']);newcat=dict(config['catalogue'])
    for key in ('population_path','population_sha256'):oldcat.pop(key);newcat.pop(key)
    need(oldcat==newcat
         and oldconfig['execution_templates']==config['execution_templates']==[]
         and all(oldconfig[k]==config[k]for k in ('max_cycles','max_proof_bytes','poll_seconds'))
         and value['max_cycles']==config['max_cycles']and value['max_proof_bytes']==config['max_proof_bytes'],
         'catalogue_race_budgets_or_dispatch_authority_changed')
    _catalogue_race_template_transition(value['scheduler_policy_transition_ref'],
        value['native_owner_source_commitment_ref'],owner,owner_pins,prior,profile,oldconfig,config)
    boundary_observation=_catalogue_race_population_boundary(value['population_boundary_ref'],
        value['population_boundary_independent_review_ref'],value['external_progress_ref'],
        prior,profile,oldconfig,config)
    auth=h._document(value['prior_generation_authorization_ref']);ack=h._document(value['prior_generation_ack_ref'])
    admission=h._document(value['prior_source_admission_authorization_ref'])
    admitted=h._document(value['prior_source_admission_ack_ref'])
    need(auth['schema']==GENERATION_SCHEMA and ack['schema']=='agent-supervisor/scientific-bounded-generation-ack@1'
         and ack['handover_ref']==value['prior_generation_authorization_ref']
         and auth['generation_number']==ack['generation_number']==value['generation_number']==2
         and auth['max_generations']==ack['max_generations']==value['max_generations']==2
         and auth['global_max_bytes']==GLOBAL_RETAINED_BYTES
         and admission['schema']==ADMISSION_SCHEMA and admitted['schema']==ADMISSION_ACK_SCHEMA
         and admitted['recovery_ref']==value['prior_source_admission_authorization_ref']
         and admission['prior_generation_authorization_ref']==value['prior_generation_authorization_ref']
         and admission['prior_generation_ack_ref']==value['prior_generation_ack_ref']
         and admitted['consuming_main_identity']==value['prior_main_identity']
         and admission['profile_ref']==value['prior_profile_ref']
         and admission['spent_cycle_count']==admitted['spent_cycle_count']
         and admitted['next_cycle_number']==admission['spent_cycle_count']+1
         and admission['generation_number']==admission['max_generations']==2
         and admitted['generation_number']==admitted['max_generations']==2,
         'catalogue_race_prior_consumed_authority_changed')
    root=Path(profile['proof_root']);spent=value['spent_cycle_count'];base=ack['spent_cycle_count']
    need(h._integer(spent,admission['spent_cycle_count']+1,config['max_cycles']-1)
         and value['prior_source_admission_ack_ref']['path']==str(root/'source-admission-reconciliation.json')
         and value['prior_generation_ack_ref']['path']==str(root/'bounded-generation-handover.json')
         and value['global_hold_ref']==admitted['retained_global_hold_ref'],
         'catalogue_race_counter_or_authority_path_changed')
    need(h._integer(base,1)and h._integer(admission['spent_cycle_count'],base+1,spent-1),
         'catalogue_race_counter_types')
    h._bound(value['global_hold_ref'])
    prior_identity=value['prior_main_identity'];h._stored_identity(prior_identity,value['prior_profile_ref'],prior)
    need(h._definitely_dead_identity(prior_identity,dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())),
         'catalogue_race_prior_Main_live_or_unknown')
    for ref in auth['retained_generations']:_generation_census(ref)
    refs=value['local_cycle_census_refs'];need(type(refs)is list and len(refs)==spent-base,
        'catalogue_race_complete_local_cycle_census_required')
    censuses={row['root']:row for row in (_generation_census(ref)for ref in refs)}
    cycles={};identities={}
    older=h._document(admission['prior_profile_snapshot_ref'])
    old_identity=admission['prior_main_identity']
    for cycle in root.glob('cycle-*'):
        intent=h._document(h.descriptor(cycle/'intent.json'));number=intent.get('cycle_number',intent.get('sequence'))
        need(h._integer(number,base+1,spent)and number not in cycles
             and cycle.name.startswith('cycle-%08d-'%number)and str(cycle)in censuses,
             'catalogue_race_cycle_number_gap_or_alias')
        cycles[number]=(cycle,intent)
    need(len(censuses)==len(refs)==len(cycles)and set(cycles)==set(range(base+1,spent+1)),
         'catalogue_race_spent_history_reset')
    for number,(cycle,intent)in cycles.items():
        need(intent['kind']=='genuine_scoped_idle'and intent['automatic_retry_admitted']is False,
             'catalogue_race_non_idle_history')
        if number==admission['spent_cycle_count']:
            need(admission['failed_intent_ref']==h.descriptor(cycle/'intent.json')
                 and admission['failed_request_ref']==intent['request_ref']
                 and admission['failed_stdout_ref']==h.descriptor(cycle/'owner.stdout')
                 and admitted['retained_cycle_hold_ref']==h.descriptor(cycle/'cycle-held.json')
                 and not (cycle/'closed.json').exists(),'catalogue_race_prior_source_negative_changed')
            continue
        if number==spent:continue
        closed=h._document(h.descriptor(cycle/'closed.json'));proof_ref=closed['cycle_proof_ref'];proof=h._document(proof_ref)
        selected,identity=(older,old_identity)if number<admission['spent_cycle_count']else(prior,prior_identity)
        need(proof_ref['path']==str(cycle/'cycle-proof.json')and proof['request_ref']==intent['request_ref']
             and proof['sequence']==intent['sequence'],'catalogue_race_historical_cycle_join')
        native=h._document(proof['native_diagnostic_ref'])
        h._cycle(proof_ref,selected,identity,h._native(native,selected,now=proof['finished_at']),
            now=proof['finished_at'],monotonic_ns=proof['finished_monotonic_ns'])
    failed,intent=cycles[spent];proof_ref=value['failed_cycle_proof_ref'];proof=h._document(proof_ref)
    need(proof_ref['path']==str(failed/'cycle-proof.json')and proof['sequence']==intent['sequence']
         and proof['request_ref']==intent['request_ref']and not (failed/'closed.json').exists()
         and value['failed_cycle_hold_ref']==h.descriptor(failed/'cycle-held.json'),
         'catalogue_race_latest_negative_cycle_join')
    need(h._definitely_dead_identity(proof['owner_observation'],
        dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())),
        'catalogue_race_prior_owner_live_or_unknown')
    hold=h._document(value['failed_cycle_hold_ref'])
    need(hold.get('schema')=='agent-supervisor/scientific-cycle-held@1'
         and hold.get('reason')in CATALOGUE_RACE_REJECTIONS and hold.get('automatic_retry_admitted')is False,
         'catalogue_race_not_exact_original_hold')
    recognition=closed_catalogue_race_rejection(proof_ref,prior,prior_identity)
    need(recognition['rejection']==hold['reason'],'catalogue_race_original_hold_reason_differs')
    previous=h._document(value['prior_health_receipt_ref'])
    need(value['prior_health_receipt_ref']['path']==prior['receipt_path']
         and previous['cycle_proof_ref']['path']==str(cycles[spent-1][0]/'cycle-proof.json')
         and previous['sequence']==cycles[spent-1][1]['sequence']==proof['sequence']-1,
         'catalogue_race_last_positive_receipt_not_current_history')
    n=h._document(h._document(previous['cycle_proof_ref'])['native_diagnostic_ref'])
    h._receipt(value['prior_health_receipt_ref'],value['prior_profile_ref'],prior,
        h._native(n,prior,now=previous['observed_at']),prior_identity,
        now=previous['observed_at'],monotonic_ns=previous['observed_monotonic_ns'])
    negative=h._document(value['negative_audit_ref'])
    need(negative.get('schema')=='scientific-closed-catalogue-race-independent-negative/v1'
         and negative.get('accepted_as_closed_negative')is True
         and negative.get('accepted_as_runtime_health')is False
         and negative.get('cycle_proof_ref')==proof_ref
         and negative.get('recognition')==recognition
         and negative.get('spent_cycle_count')==spent
         and negative.get('prior_source_admission_ack_ref')==value['prior_source_admission_ack_ref']
         and negative.get('recovery_or_retry_authority')is False,
         'catalogue_race_independent_negative_join')
    _catalogue_race_external_progress(value['external_progress_ref'],recognition,proof,
        h._document(proof['result_ref']),boundary_observation=boundary_observation)
    request=h._document(proof['request_ref'])
    _catalogue_race_scheduler_transition(request,value['scheduler_snapshot_ref'],owner['scheduler_policy_transition'])
    _released_initialization_record(request,h._document(proof['terminal_ref']))
    fresh=h._document(value['current_native_ref']);h._native(fresh,profile)
    _catalogue_race_current_progress(boundary_observation,recognition,profile,fresh)
    need(select_template(config,profile,fresh)[1]=='genuine_scoped_idle',
         'catalogue_race_current_scope_not_quiet')
    catalogue_race_source_inventory(value['source_commitment_ref'],profile,catalogue_parser_reference(profile))
    check_proof_budget(root,config['max_proof_bytes'],reserve_bytes=REUSE_CYCLE_RESERVE_BYTES,
        reserve_files=REUSE_CYCLE_RESERVE_FILES)
    for field in ('prior_profile_snapshot_ref','prior_config_snapshot_ref','negative_audit_ref',
                  'external_progress_ref','population_boundary_ref','population_boundary_independent_review_ref','current_native_ref','source_commitment_ref',
                  'scheduler_snapshot_ref','native_owner_source_commitment_ref','scheduler_policy_transition_ref'):
        need(not Path(value[field]['path']).lstat().st_mode&0o222,'catalogue_race_control_snapshot_not_frozen')
    return spent,list(auth['retained_generations'])


def authorize_closed_catalogue_race_recovery(spec_ref,output):
    """Issue a fresh explicit one-use grant; do not start, replay or clear holds."""
    h=health_owner();spec=h._document(spec_ref)
    h._keys(spec,CATALOGUE_RACE_KEYS-{'recorded_at','manager_before'},'catalogue_race_authorization_spec')
    profile=h._profile(spec['profile_ref']);argv=profile['unit']['argv']
    config_pin=next(p for p in profile['source_descriptors']if p['path']==argv[argv.index('--config')+1])
    config,profile_ref,_=load_configuration(config_pin);root=Path(profile['proof_root'])
    need(not os.path.lexists(root/'catalogue-race-reconciliation-v01.json'),
         'catalogue_race_authorization_already_consumed')
    with os.fdopen(os.open(root/'scientific-executor.lock',os.O_RDWR|os.O_NOFOLLOW),'rb')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manager=h._manager(profile['unit']['name'])
        need(manager.get('MainPID')=='0'and manager.get('ActiveState')in ('failed','inactive'),
             'catalogue_race_admin_requires_no_Main')
        value=dict(spec,recorded_at=time.time(),manager_before=manager)
        _validate_closed_catalogue_race_recovery(value,profile_ref,profile,config)
        output=h._path(str(output));need(root in output.parents,'catalogue_race_grant_outside_same_root')
        reference=put(output,value);output.chmod(0o400);return reference


def reconcile_closed_catalogue_race_recovery(reference,profile_ref,profile,config,opening_identity):
    h=health_owner();value=h._document(reference);argv=profile['unit']['argv']
    flag='--known-closed-catalogue-race-recovery'
    need(not os.path.lexists(Path(profile['proof_root'])/'catalogue-race-reconciliation-v01.json'),
         'catalogue_race_authorization_already_consumed')
    need(os.environ.get(CATALOGUE_RACE_SHA_ENV)==reference['sha256']
         and argv.count(flag)==1 and argv[argv.index(flag)+1]==reference['path']
         and not any(x in argv for x in ('--known-nonstarted-recovery','--known-closed-idle-recovery',
             '--bounded-generation-handover','--known-source-admission-recovery'))
         and not Path(reference['path']).lstat().st_mode&0o222,'catalogue_race_grant_not_unit_pinned')
    spent,censuses=_validate_closed_catalogue_race_recovery(value,profile_ref,profile,config)
    ack=dict(schema=CATALOGUE_RACE_ACK_SCHEMA,recovery_ref=reference,
        consuming_main_identity=opening_identity,spent_cycle_count=spent,next_cycle_number=spent+1,
        generation_number=2,max_generations=2,retained_global_hold_ref=value['global_hold_ref'],
        retained_failed_cycle_hold_ref=value['failed_cycle_hold_ref'],
        prior_source_admission_ack_ref=value['prior_source_admission_ack_ref'],
        initial_consecutive_catalogue_race_failures=1,no_health_or_task_credit_for_failed_cycle=True)
    put(Path(profile['proof_root'])/'catalogue-race-reconciliation-v01.json',ack)
    return spent,censuses



def _validate_source_admission_recovery(value, profile_ref, profile, config):
    """Same-root producer repair; preserve consumed generation and physical cycles."""
    h=health_owner();h._keys(value,ADMISSION_KEYS,'source_admission_recovery_shape')
    need(value['schema']==ADMISSION_SCHEMA and value['authorization']=='resume_idle_after_closed_source_admission_timeout'
         and value['profile_ref']==profile_ref and h._profile(profile_ref)==profile,'source_admission_recovery_authority')
    now=time.time()
    need(h._number(value['recorded_at'])and 0<=now-value['recorded_at']<=profile['max_age_seconds']
         and type(value['manager_before'])is dict and value['manager_before'].get('Id')==profile['unit']['name']
         and value['manager_before'].get('MainPID')=='0'
         and value['manager_before'].get('ActiveState')in('failed','inactive'),'source_admission_admin_clock_or_manager')
    for field in ('prior_profile_snapshot_ref','prior_config_snapshot_ref','negative_audit_ref','scheduler_snapshot_ref','current_native_ref','source_commitment_ref'):
        need(not Path(value[field]['path']).lstat().st_mode&0o222,'source_admission_control_snapshot_not_frozen')
    prior=h._document(value['prior_profile_snapshot_ref']);oldconfig=h._document(value['prior_config_snapshot_ref'])
    need(_same_payload(value['prior_profile_snapshot_ref'],value['prior_profile_ref'])
         and profile['proof_root']==prior['proof_root'] and profile['receipt_path']!=prior['receipt_path']
         and Path(profile['proof_root']) in Path(profile['receipt_path']).parents
         and not os.path.lexists(profile['receipt_path']),'source_admission_same_root_new_health_epoch')
    for key in ('catalogue','task_scope','max_age_seconds','max_cycle_seconds','cycle_contract'):
        need(profile[key]==prior[key],'source_admission_scope_changed')
    for role in ('health_module','owner_source','driver_source'):
        need(profile[role]==prior[role],'source_admission_backend_or_verifier_descriptor_changed')
    need(all(pin in profile['source_descriptors']for pin in prior['source_descriptors']),
         'source_admission_original_native_source_pin_removed')
    need(profile['unit']['name']==prior['unit']['name']and profile['unit']['interpreter']==prior['unit']['interpreter']
         and profile['unit']['working_directory']==prior['unit']['working_directory'], 'source_admission_unit_scope_changed')
    oldargv=prior['unit']['argv'];oldconfigpath=oldargv[oldargv.index('--config')+1]
    oldconfigpin=next(pin for pin in prior['source_descriptors']if pin['path']==oldconfigpath)
    need(_same_payload(value['prior_config_snapshot_ref'],oldconfigpin)and oldconfigpin['sha256']==oldargv[oldargv.index('--config-sha256')+1], 'source_admission_prior_configuration_snapshot')
    need(source_admission_unit_projection(profile['unit']['argv'],profile['main_entrypoint']['path'])==source_admission_unit_projection(prior['unit']['argv'],prior['main_entrypoint']['path']),'source_admission_unit_command_broadened')
    need(profile['unit']['environment']==prior['unit']['environment'],'source_admission_unit_environment_changed')
    need(oldconfig['execution_templates']==config['execution_templates']==[]
         and oldconfig['idle_template_ref']==config['idle_template_ref']
         and all(config[k]==oldconfig[k]for k in ('catalogue','max_cycles','max_proof_bytes','poll_seconds'))
         and value['max_cycles']==config['max_cycles']and value['max_proof_bytes']==config['max_proof_bytes'],
         'source_admission_budgets_or_task_authority_changed')
    auth=h._document(value['prior_generation_authorization_ref']);ack=h._document(value['prior_generation_ack_ref'])
    need(auth['schema']==GENERATION_SCHEMA and auth['generation_number']==ack['generation_number']==value['generation_number']==2
         and auth['max_generations']==ack['max_generations']==value['max_generations']==2
         and ack['schema']=='agent-supervisor/scientific-bounded-generation-ack@1'
         and ack['handover_ref']==value['prior_generation_authorization_ref']
         and value['prior_generation_ack_ref']['path']==str(Path(profile['proof_root'])/'bounded-generation-handover.json')
         and ack['spent_cycle_count']==auth['spent_cycle_count']and ack['next_cycle_number']==auth['spent_cycle_count']+1
         and auth['profile_ref']==value['prior_profile_ref']and ack['consuming_main_identity']==value['prior_main_identity'],
         'source_admission_consumed_generation_changed')
    need(oldargv.count('--bounded-generation-handover')==1 and oldargv[oldargv.index('--bounded-generation-handover')+1]==value['prior_generation_authorization_ref']['path']
         and type(auth['retained_generations'])is list and len(auth['retained_generations'])==1,'source_admission_prior_generation_authority_join')
    for ref in auth['retained_generations']:_generation_census(ref)
    oldtemplate=h._document(config['idle_template_ref']);oldplan=h._document(oldtemplate['plan_ref'])
    need(oldplan['native_metadata_cli'].get('schema')=='native-embedded-scoped-observation-cli/v3'
         and oldplan['native_metadata_cli'].get('source_capsule_ref')==auth['source_capsule_ref']
         and auth['source_capsule_ref']in profile['source_descriptors'],'source_admission_original_cache_binding_changed')
    need(auth['global_max_bytes']==GLOBAL_RETAINED_BYTES,'source_admission_global_budget_changed')
    root=Path(profile['proof_root']);base=ack['spent_cycle_count'];spent=value['spent_cycle_count']
    need(h._integer(base,1)and h._integer(spent,base+1,config['max_cycles']-1),'source_admission_spent_count')
    cycles=list(root.glob('cycle-*'));numbers={}
    for cycle in cycles:
        intent=h._document(h.descriptor(cycle/'intent.json'));number=intent.get('cycle_number',intent.get('sequence'))
        need(h._integer(number,base+1,spent)and cycle.name.startswith('cycle-%08d-'%number)and number not in numbers,
             'source_admission_cycle_number_changed')
        numbers[number]=(cycle,intent)
    need(set(numbers)==set(range(base+1,spent+1))and len(cycles)+base==spent,'source_admission_spent_history_reset_or_gap')
    old_identity=value['prior_main_identity'];h._stored_identity(old_identity,value['prior_profile_ref'],prior)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(old_identity['boot_id']==boot and source_admission_PID_absent(old_identity['pid']),
         'source_admission_original_Main_PID_not_absent_same_boot')
    for number,(cycle,intent)in numbers.items():
        need(intent.get('kind')=='genuine_scoped_idle'and intent.get('automatic_retry_admitted')is False,'source_admission_non_idle_history')
        if number==spent:continue
        closed=h._document(h.descriptor(cycle/'closed.json'));proof_ref=closed['cycle_proof_ref'];proof=h._document(proof_ref)
        need(proof_ref['path']==str(cycle/'cycle-proof.json')and proof['request_ref']==intent['request_ref']
             and proof['sequence']==intent['sequence'],'source_admission_historical_proof_join')
        native=h._document(proof['native_diagnostic_ref'])
        h._cycle(proof_ref,prior,old_identity,h._native(native,prior,now=proof['finished_at']),
                 now=proof['finished_at'],monotonic_ns=proof['finished_monotonic_ns'])
    need(value['prior_health_receipt_ref']['path']==prior['receipt_path'],'source_admission_prior_health_path')
    receipt=h._document(value['prior_health_receipt_ref']);need(receipt['cycle_proof_ref']['path']==str(numbers[spent-1][0]/'cycle-proof.json')
         and receipt['sequence']==numbers[spent-1][1]['sequence']==numbers[spent][1]['sequence']-1,'source_admission_latest_health_cycle_join')
    native=h._document(h._document(receipt['cycle_proof_ref'])['native_diagnostic_ref'])
    h._receipt(value['prior_health_receipt_ref'],value['prior_profile_ref'],prior,h._native(native,prior,now=receipt['observed_at']),
               old_identity,now=receipt['observed_at'],monotonic_ns=receipt['observed_monotonic_ns'])
    failed,intent=numbers[spent]
    need(value['failed_intent_ref']['path']==str(failed/'intent.json')and h._document(value['failed_intent_ref'])==intent
         and value['failed_request_ref']==intent['request_ref']and value['failed_stdout_ref']['path']==str(failed/'owner.stdout'),
         'source_admission_failed_cycle_artifact_join')
    request=h._document(value['failed_request_ref']);terminal=original_terminal(h._bound(value['failed_stdout_ref']),value['failed_request_ref'])
    source_admission_terminal(terminal,value['failed_request_ref'],request,failed)
    need(request['source_sha256'].get(prior['owner_source']['path'])==prior['owner_source']['sha256']
         and request['source_sha256'].get(prior['driver_source']['path'])==prior['driver_source']['sha256']
         and request['source_roots']==[str(ORIGINAL_NAMED_ROOTS[0].parents[1]),'/home/barberb/lift_coding/external/ipfs_accelerate'],
         'source_admission_original_owner_or_source_roots_changed')
    need(value['global_hold_ref']['path']==str(root/'executor-held.json')and value['cycle_hold_ref']['path']==str(failed/'cycle-held.json'),
         'source_admission_hold_paths_changed')
    global_hold=h._document(value['global_hold_ref']);cycle_hold=h._document(value['cycle_hold_ref'])
    need(global_hold.get('cycle_root')==str(failed)and global_hold.get('reason')=='original_owner_failed'
         and cycle_hold.get('reason')=='original_owner_failed'and global_hold.get('automatic_retry_admitted')is False
         and cycle_hold.get('automatic_retry_admitted')is False,'source_admission_original_holds_changed')
    audit=h._document(value['negative_audit_ref']);need(audit.get('schema')=='scientific-nonstarted-source-admission-independent-negative/v1'
         and audit.get('accepted_as_known_nonstarted_negative')is True and audit.get('accepted_as_runtime_health')is False
         and audit.get('no_child_native_storage_or_control_effects')is True and audit.get('attempt_directory_empty')is True
         and audit.get('same_boot_literal_Main_PID_absence_verified')is True
         and audit.get('same_boot_literal_owner_PID_absence_verified')is True and audit.get('owner_birth_not_present_or_fabricated')is True
         and audit.get('prior_profile_ref')==value['prior_profile_ref']
         and audit.get('prior_health_receipt_ref')==value['prior_health_receipt_ref']
         and audit.get('prior_generation_authorization_ref')==value['prior_generation_authorization_ref']
         and audit.get('prior_generation_ack_ref')==value['prior_generation_ack_ref']
         and audit.get('spent_cycle_count')==spent and audit.get('historical_carried_cycle_count')==base
         and audit.get('local_cycle_count')==len(cycles)and audit.get('recovery_or_retry_authority')is False
         and audit.get('request_ref')==value['failed_request_ref']and audit.get('owner_stdout_ref')==value['failed_stdout_ref']
         and audit.get('global_hold_ref')==value['global_hold_ref']and audit.get('cycle_hold_ref')==value['cycle_hold_ref'],
         'source_admission_independent_negative_artifact_join')
    _recovery_scheduler(request,value['scheduler_snapshot_ref'])
    fresh=h._document(value['current_native_ref']);h._native(fresh,profile)
    need(select_template(config,profile,fresh)[1]=='genuine_scoped_idle','source_admission_current_scope_not_quiet')
    source_admission_source_inventory(value['source_commitment_ref'],profile)
    check_proof_budget(root,config['max_proof_bytes'],reserve_bytes=REUSE_CYCLE_RESERVE_BYTES,reserve_files=REUSE_CYCLE_RESERVE_FILES)
    return spent


def authorize_source_admission_recovery(spec_ref, output):
    """One explicit source-qualified administrative grant; no start or rearm."""
    h=health_owner();spec=h._document(spec_ref);h._keys(spec,ADMISSION_KEYS-{'recorded_at','manager_before'},'source_admission_spec')
    profile=h._profile(spec['profile_ref']);config_pin=next(pin for pin in profile['source_descriptors']if pin['path']==profile['unit']['argv'][profile['unit']['argv'].index('--config')+1])
    config,profile_ref,_=load_configuration(config_pin);manager=h._manager(profile['unit']['name'])
    need(manager.get('MainPID')=='0'and manager.get('ActiveState')in('failed','inactive'),'source_admission_admin_requires_no_Main')
    value=dict(spec,recorded_at=time.time(),manager_before=manager);_validate_source_admission_recovery(value,profile_ref,profile,config)
    output=h._path(str(output));need(Path(profile['proof_root'])in output.parents,'source_admission_grant_outside_same_root')
    ref=put(output,value);output.chmod(0o400);return ref


def reconcile_source_admission_recovery(reference, profile_ref, profile, config, opening_identity):
    h=health_owner();value=h._document(reference)
    need(os.environ.get(ADMISSION_SHA_ENV)==reference['sha256'],'source_admission_grant_not_unit_pinned')
    argv=profile['unit']['argv'];flag='--known-source-admission-recovery'
    need(argv.count(flag)==1 and argv[argv.index(flag)+1]==reference['path']
         and not any(x in argv for x in ('--known-nonstarted-recovery','--known-closed-idle-recovery','--bounded-generation-handover'))
         and not Path(reference['path']).lstat().st_mode&0o222,'source_admission_grant_argv_or_mode')
    spent=_validate_source_admission_recovery(value,profile_ref,profile,config)
    acknowledgement=dict(schema=ADMISSION_ACK_SCHEMA,recovery_ref=reference,consuming_main_identity=opening_identity,
        retained_global_hold_ref=value['global_hold_ref'],retained_cycle_hold_ref=value['cycle_hold_ref'],spent_cycle_count=spent,
        next_cycle_number=spent+1,generation_number=value['generation_number'],max_generations=value['max_generations'],
        initial_consecutive_admission_failures=1,no_health_or_task_credit_for_failed_cycle=True)
    put(Path(profile['proof_root'])/'source-admission-reconciliation.json',acknowledgement)
    return spent,1


def health_owner():
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health
    return native_executor_health


def need(value, reason):
    if not value:
        raise ExecutorHold(reason)


def put(path, value):
    """Create one immutable private artifact; never replace an existing one."""
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    need(len(raw) <= 4 * 1024**2, 'artifact_bound')
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return health_owner().descriptor(path)


def load_configuration(config_ref):
    h = health_owner()
    value = h._document(config_ref)
    h._keys(value, CONFIG_KEYS, 'service_config_shape')
    need(value['schema'] == CONFIG_SCHEMA, 'service_config_schema')
    profile_ref = h.descriptor(h._path(value['profile_path']))
    profile = h._profile(profile_ref)
    need(config_ref in profile['source_descriptors'], 'service_config_not_source_pinned')
    h._keys(value['catalogue'], CATALOGUE_KEYS, 'service_catalogue_shape')
    catalogue = value['catalogue']
    h._path(catalogue['database_path']); h._path(catalogue['population_path'])
    need(catalogue['board_id'] == 'legal-ir-scientific', 'service_board_scope')
    need(all(catalogue[k] == profile['catalogue'][k] for k in
        ('board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256')), 'service_catalogue_binding')
    need(h._integer(catalogue['max_age_seconds'], 1, profile['max_age_seconds']) and
        h._integer(catalogue['timeout_seconds'], 1, min(30, profile['max_cycle_seconds'])), 'service_observation_bounds')
    need(h._integer(value['poll_seconds'], 1, 60) and h._integer(value['max_cycles'], 2, 10000)
        and h._integer(value['max_proof_bytes'], 1024**2, 16 * 1024**3), 'service_finite_limits')
    idle = h._document(value['idle_template_ref'])
    validate_template(idle, profile, 'genuine_scoped_idle')
    templates = value['execution_templates']
    need(type(templates) is list and len(templates) <= len(profile['task_scope']), 'service_execution_population')
    seen = set()
    for row in templates:
        h._keys(row, {'task_cid', 'expected_revision', 'template_ref'}, 'service_execution_template_shape')
        need(row['task_cid'] in profile['task_scope'] and row['task_cid'] not in seen and
            h._integer(row['expected_revision'], 1), 'service_execution_binding')
        seen.add(row['task_cid'])
        validate_template(h._document(row['template_ref']), profile, 'native_execution')
    return value, profile_ref, profile


def validate_template(template, profile, kind):
    h = health_owner()
    h._keys(template, TEMPLATE_KEYS, 'cycle_template_shape')
    need(template['schema'] == TEMPLATE_SCHEMA and template['kind'] == kind, 'cycle_template_kind')
    seed = h._path(template['seed_root'])
    need(Path(profile['proof_root']) in seed.parents, 'seed_outside_proof_root')
    for field in ('request_ref', 'plan_ref'):
        h._bound(template[field]); need(seed in Path(template[field]['path']).parents, 'template_artifact_outside_seed')
    h._keys(template['links'], {'execution_request', 'result', 'origins'}, 'cycle_template_links')
    for value in template['links'].values():
        need(type(value) is str and value and not Path(value).is_absolute() and '..' not in Path(value).parts,
            'template_relative_link')
    need(len(set(template['links'].values())) == 3, 'template_link_alias')


def observe_native(config, *, catalogue_parser=None):
    if catalogue_parser is None:
        from ipfs_accelerate_py.agent_supervisor.rescue.native_catalogue_probe import observe_native_catalogue
    else:
        need(type(catalogue_parser)is _ExplicitCatalogueParser,'catalogue_parser_unselected_callable')
        h=health_owner();raw=h._bound(catalogue_parser.descriptor)
        observe_native_catalogue=catalogue_parser.observer
        need(catalogue_parser.module.__dict__.get('observe_native_catalogue')is observe_native_catalogue
             and observe_native_catalogue.__globals__ is catalogue_parser.module.__dict__
             and observe_native_catalogue.__code__.co_filename==catalogue_parser.descriptor['path'],
             'catalogue_parser_callback_source_changed')
    binding = dict(config['catalogue'])
    database, population = Path(binding.pop('database_path')), Path(binding.pop('population_path'))
    report = observe_native_catalogue(database, population, **binding)
    if catalogue_parser is not None:
        need(h._bound(catalogue_parser.descriptor)==raw,'catalogue_parser_source_changed_during_observation')
    value = report.get('details', {}).get('native_catalogue')
    need(type(value) is dict, 'native_observation_missing')
    return value


def select_template(config, profile, native):
    """Never interpret zero globally ready tasks as authority to ignore scope."""
    health_owner()._native(native, profile)
    tasks = {row['task_cid']: row for row in native['tasks']}
    scoped = [tasks[cid] for cid in profile['task_scope']]
    if all(row['status'] in QUIET_STATUSES for row in scoped):
        return config['idle_template_ref'], 'genuine_scoped_idle'
    permitted = {(row['task_cid'], row['expected_revision']): row['template_ref']
                 for row in config['execution_templates']}
    for row in sorted(scoped, key=lambda item: item['task_cid']):
        if row['status'] == 'ready' and (row['task_cid'], row['revision']) in permitted:
            return permitted[(row['task_cid'], row['revision'])], 'native_execution'
    raise ExecutorHold('scoped_work_has_no_explicit_execution_binding')


def relocate(value, seed, root):
    """Rewrite only paths rooted in the sealed seed; preserve external pins."""
    if isinstance(value, str):
        if value == str(seed) or value.startswith(str(seed) + os.sep):
            return str(root) + value[len(str(seed)):]
        return value
    if type(value) is list:
        return [relocate(item, seed, root) for item in value]
    if type(value) is dict:
        return {relocate(key, seed, root): relocate(item, seed, root) for key, item in value.items()}
    return value


def check_proof_budget(root, maximum, *, reserve_bytes=0, reserve_files=0):
    need(type(reserve_bytes)is int and reserve_bytes>=0 and type(reserve_files)is int and reserve_files>=0, 'proof_reserve_shape')
    total = count = 0
    for path in root.rglob('*'):
        info = path.lstat()
        need(not stat.S_ISLNK(info.st_mode), 'proof_symlink')
        if stat.S_ISDIR(info.st_mode):
            continue
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'proof_nonordinary_artifact')
        total += info.st_size; count += 1
        need(count <= 20000 and total <= maximum, 'proof_storage_bound')
    need(count+reserve_files <= 20000 and total+reserve_bytes <= maximum, 'proof_projected_storage_bound')
    return {'files':count,'bytes':total}


def prepare_cycle(template_ref, kind, config, profile, root):
    h = health_owner(); template = h._document(template_ref)
    validate_template(template, profile, kind)
    seed = Path(template['seed_root'])
    plan = relocate(h._document(template['plan_ref']), seed, root)
    request = relocate(h._document(template['request_ref']), seed, root)
    need(plan.get('schema') == profile['cycle_contract']['plan_schema'] and
        plan.get('operation') == profile['cycle_contract']['plan_operation'], 'prepared_plan_contract')
    if kind == 'genuine_scoped_idle':
        need(plan.get('task_scope') == profile['task_scope'], 'prepared_idle_scope')
    else:
        need(plan.get('task_cid') in profile['task_scope'] and any(row['task_cid'] == plan.get('task_cid')
            and row['expected_revision'] == plan.get('expected_ready_revision')
            and row['template_ref'] == template_ref for row in config['execution_templates']), 'prepared_execution_scope')
    need(request.get('schema') == 'legal-ir-original-native-control-request/v1' and
        request.get('purpose') == 'native_control_metadata', 'prepared_original_request_schema')
    need(h._integer(request.get('timeout_seconds'), 1, profile['max_cycle_seconds']), 'prepared_owner_timeout')
    sources = request.get('source_sha256')
    need(type(sources) is dict and all(sources.get(profile[field]['path']) == profile[field]['sha256']
        for field in ('owner_source', 'driver_source')), 'prepared_original_source_binding')
    # Refresh only the read-only DB preimage. Nothing changes task admission.
    preimage = request.get('database_preimage')
    need(type(preimage) is dict and preimage.get('path') == config['catalogue']['database_path'], 'prepared_database_binding')
    current_db = h.database_descriptor(config['catalogue']['database_path'])
    request['database_preimage'] = current_db
    command = plan.get('native_metadata_cli', {})
    if kind == 'genuine_scoped_idle':
        binding = command.get('observation_binding', {})
        need(binding.get('database_preimage', {}).get('path') == current_db['path'], 'prepared_observation_database')
        binding['database_preimage'] = current_db
    attempt = h._path(request['attempt_directory'])
    need(root in attempt.parents and not attempt.exists(), 'prepared_attempt_directory')
    attempt.mkdir(mode=0o700, parents=True)
    plan_ref = put(root / 'operation-plan.json', plan)
    old_plan = str(root / Path(template['plan_ref']['path']).relative_to(seed))
    sources.pop(old_plan, None); sources[plan_ref['path']] = plan_ref['sha256']
    request['argv'] = [profile['unit']['argv'][0], '-B', profile['driver_source']['path'],
        '--plan', plan_ref['path'], '--plan-sha256', plan_ref['sha256']]
    request['request_id'] = 'scientific-executor:' + root.name
    request_ref = put(root / 'request.json', request)
    links = {key: root / value for key, value in template['links'].items()}
    need(command.get('result_path') == str(links['result']), 'prepared_result_link')
    return request_ref, plan_ref, links


def original_terminal(raw, request_ref):
    h = health_owner(); candidates = []
    try:
        candidates.append(h._decode(raw))
    except (ValueError, TypeError):
        pass
    for line in raw.splitlines():
        try:
            candidate = h._decode(line)
        except (ValueError, TypeError):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    matches = [row for row in candidates if type(row) is dict and
        row.get('schema') == 'legal-ir-original-native-control-terminal/v1' and
        row.get('request_sha256') == request_ref['sha256']]
    need(len(matches) == 1, 'original_terminal_missing_or_ambiguous')
    return matches[0]


def persistent_hold(proof_root, cycle_root, reason):
    path = Path(proof_root) / 'executor-held.json'
    if not os.path.lexists(path):
        put(path, {'schema': 'agent-supervisor/scientific-executor-persistent-hold@1',
            'cycle_root': str(cycle_root), 'reason': str(reason)[:2048],
            'recorded_at': time.time(), 'automatic_retry_admitted': False})


def _same_payload(left, right):
    """A durable historical snapshot keeps bytes/hash, not its temporary name."""
    return type(left) is dict and type(right) is dict and all(left.get(key) == right.get(key) for key in ('bytes', 'sha256'))


def _recovery_scheduler(request, snapshot_ref):
    """Read the original state; never allocate, normalize, prune or persist it."""
    h = health_owner(); expected = request['expected_scheduler_policy_sha256']
    family = {request['request_id'], request['request_id'] + ':root', request['request_id'] + ':phase'}
    archived = h._document(snapshot_ref)
    # The live scheduler is operational DATA; group-write permission is not
    # executable authority. The archived snapshot retains strict control rules.
    path = Path(request['scheduler_state_path']); before = path.lstat()
    need(0 < before.st_size <= h.MAX_JSON, 'recovery_scheduler_data_bound')
    live_ref = h.database_descriptor(path)
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid,
                       s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        first = os.fstat(stream.fileno()); raw = stream.read(h.MAX_JSON + 1)
        last = os.fstat(stream.fileno())
    need(stamp(before) == stamp(first) == stamp(last) == stamp(path.lstat())
         and len(raw) == live_ref['bytes'] <= h.MAX_JSON
         and hashlib.sha256(raw).hexdigest() == live_ref['sha256']
         and h.database_descriptor(path) == live_ref, 'recovery_scheduler_data_changed')
    for state in (archived, h._decode(raw)):
        need(type(state.get('config')) is dict, 'recovery_scheduler_policy_missing')
        digest = hashlib.sha256(json.dumps(state['config'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        need(digest == expected, 'recovery_scheduler_policy_changed')
        for field in ('waiters', 'leases'):
            rows = state.get(field)
            need(type(rows) is dict and all(type(row) is dict for row in rows.values()), 'recovery_scheduler_rows_unknown')
            need(not any(row.get('request_id') in family for row in rows.values()), 'recovery_original_request_still_active')


def _validate_nonstarted_recovery(value, profile_ref, profile, config):
    """Closed, idle-only negative reconciliation; this cannot establish health."""
    h = health_owner(); h._keys(value, RECOVERY_KEYS, 'recovery_shape')
    need(h._profile(profile_ref) == profile, 'recovery_current_profile_changed')
    need(value['schema'] == RECOVERY_SCHEMA and value['authorization'] == 'resume_genuine_scoped_idle'
         and value['profile_ref'] == profile_ref and value['max_cycles'] == config['max_cycles']
         and config['execution_templates'] == [], 'recovery_profile_or_authority')
    root = Path(profile['proof_root']); archive = root / 'reconciliation-v01'
    total = 0
    for field in RECOVERY_ARCHIVE_FIELDS:
        h._within(value[field], archive); h._bound(value[field]); total += value[field]['bytes']
    need(total <= 1024**2, 'recovery_archive_bound')
    prior = h._document(value['prior_profile_snapshot_ref'])
    need(_same_payload(value['prior_profile_ref'], value['prior_profile_snapshot_ref'])
         and value['prior_profile_ref']['path'] == profile_ref['path'], 'recovery_prior_profile_snapshot')
    for field in ('catalogue', 'task_scope', 'proof_root', 'receipt_path', 'max_age_seconds', 'max_cycle_seconds',
                  'health_module', 'owner_source', 'driver_source'):
        need(prior.get(field) == profile[field], 'recovery_prior_profile_scope')
    need(all(prior['unit'][key] == profile['unit'][key] for key in
         ('name', 'interpreter', 'working_directory', 'environment')), 'recovery_prior_unit_scope')
    need(prior['main_entrypoint']['path'] == profile['main_entrypoint']['path'], 'recovery_main_source_path')
    source = h._document(value['source_commitment_ref'])
    source_root = Path(profile['owner_source']['path']).parents[3]
    need(source.get('schema') == 'legal-ir-scientific-source-commitment/v1'
         and source.get('runtime_activation_or_task_authority') is False
         and source.get('source_root') == str(source_root),
         'recovery_source_commitment')
    inventory_ref = source['source_inventory']
    need(Path(profile['proof_root']).parent in Path(inventory_ref['path']).parents,
         'recovery_inventory_not_durable')
    inventory = h._document(inventory_ref)
    need(inventory.get('schema') == 'legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(key) == source.get(key) for key in ('source_root', 'source_commit', 'source_tree'))
         and type(source.get('source_commit')) is str and len(source['source_commit']) == 40
         and set(source['source_commit']) <= set('0123456789abcdef')
         and type(source.get('source_tree')) is str and len(source['source_tree']) == 40
         and set(source['source_tree']) <= set('0123456789abcdef'), 'recovery_inventory_commitment_join')
    rows = inventory.get('files')
    need(type(rows) is list and 0 < len(rows) <= 4096 and inventory.get('file_count') == len(rows)
         and all(type(row) is dict and type(row.get('path')) is str for row in rows), 'recovery_inventory_shape')
    indexed = {row['path']: row for row in rows}
    need(len(indexed) == len(rows), 'recovery_inventory_duplicate_paths')
    selected = [pin for pin in profile['source_descriptors'] if Path(pin['path']).is_relative_to(source_root)
        and Path(pin['path']).suffix == '.py']
    need(all(profile[field] in selected for field in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'))
         and all(all(indexed.get(pin['path'], {}).get(key) == pin[key] for key in ('path', 'bytes', 'sha256'))
             for pin in selected), 'recovery_inventory_current_source_mismatch')
    hold = h._document(value['hold_ref']); failed = h._path(value['failed_cycle_root'])
    need(value['hold_ref']['path'] == str(root / 'executor-held.json') and failed.parent == root
         and hold.get('schema') == 'agent-supervisor/scientific-executor-persistent-hold@1'
         and hold.get('cycle_root') == str(failed) and hold.get('reason') == 'original_owner_failed'
         and hold.get('automatic_retry_admitted') is False, 'recovery_exact_original_hold')
    need(value['intent_ref']['path'] == str(failed / 'intent.json')
         and value['request_ref']['path'] == str(failed / 'request.json')
         and value['owner_stdout_ref']['path'] == str(failed / 'owner.stdout'), 'recovery_original_artifact_paths')
    intent = h._document(value['intent_ref']); request = h._document(value['request_ref'])
    need(intent.get('kind') == 'genuine_scoped_idle' and intent.get('automatic_retry_admitted') is False
         and intent.get('request_ref') == value['request_ref'] and h._integer(intent.get('sequence'), 1),
         'recovery_original_idle_intent')
    need(request.get('schema') == 'legal-ir-original-native-control-request/v1'
         and request.get('purpose') == 'native_control_metadata'
         and all(request['source_sha256'].get(prior[field]['path']) == prior[field]['sha256']
             for field in ('owner_source', 'driver_source')), 'recovery_original_owner_source')
    terminal = original_terminal(h._bound(value['owner_stdout_ref']), value['request_ref'])
    need(terminal.get('completed') is False and terminal.get('child_launch_attempted') is False
         and terminal.get('child_reaped') is False and terminal.get('child_returncode') is None
         and terminal.get('control_effects') == 'not_started'
         and terminal.get('storage_status') == 'not_requested'
         and terminal.get('original_native_compute_closed') is True and terminal.get('source_READ_closed') is True
         and terminal.get('absolute_resource_operation_deadline_closed') is True
         and terminal.get('cleanup_errors') == [] and terminal.get('received_signals') == []
         and terminal.get('error_type') == 'ResourceAdmissionDeadlineError'
         and terminal.get('task_completion_authority_granted_by_resource_owner') is False
         and not any(field in terminal for field in ('reservation', 'execution_request', 'external_precharge', 'process_spawn_receipts')),
         'recovery_not_proven_known_nonstarted')
    need(terminal.get('database_preimage_before_grants') == request.get('database_preimage')
         and request['database_preimage']['path'] == config['catalogue']['database_path'],
         'recovery_original_database_join')
    attempt = h._path(request['attempt_directory'])
    need(failed in attempt.parents and attempt.is_dir() and not any(attempt.iterdir())
         and not (failed / 'closed.json').exists() and not (failed / 'cycle-proof.json').exists(),
         'recovery_attempt_or_receipt_not_empty')
    capture = h._document(value['activation_capture_ref']); plan = h._document(value['activation_plan_ref'])
    need(capture.get('schema') == 'legal-ir-scientific-first-install-actual-capture/v1'
         and capture.get('accepted') is False and capture.get('applied') is True
         and capture.get('source_READ_closed_before_start') is True and capture.get('start_requested') is True
         and _same_payload(capture['plan'], value['activation_plan_ref'])
         and _same_payload(plan['files'][1]['after'], value['prior_profile_snapshot_ref']), 'recovery_original_activation_join')
    need(all(config[key] == plan['main_loop_limits'][key]
         for key in ('max_cycles', 'poll_seconds', 'max_proof_bytes')), 'recovery_original_budget_changed')
    old_main = capture['opening_actual_main_identity']
    h._stored_identity(old_main, value['prior_profile_ref'], prior)
    negative = h._document(value['negative_audit_ref'])
    need(negative.get('schema') == 'scientific-first-activation-independent-negative-audit/v1'
         and negative.get('accepted_as_runtime_activation') is False
         and negative.get('negative_terminal_and_hold_verified') is True
         and negative.get('attempt_directory_empty') is True and negative.get('cycle_receipts_absent') is True
         and negative.get('original_terminal') == terminal and negative.get('persistent_hold') == hold
         and negative['failed_initial_activation']['opening_identity'] == old_main, 'recovery_independent_negative_join')
    correction = h._document(value['memory_correction_ref'])
    need(correction.get('schema') == 'scientific-service-memory-context-actual-repair/v1'
         and correction.get('accepted') is True and correction.get('native_envelope_and_policy_changed') is False
         and correction.get('restart_or_cycle_replay') is False
         and correction['scientific_after'].get('Id') == profile['unit']['name']
         and correction['scientific_after'].get('MainPID') == '0'
         and correction['scientific_after'].get('MemoryHigh') == 'infinity', 'recovery_cause_not_corrected')
    dropin = correction['persistent_manager_dropin']
    h._bound({key: dropin[key] for key in ('path', 'bytes', 'sha256')})
    need(dropin.get('text') and 'MemoryHigh=infinity' in dropin['text'].splitlines(), 'recovery_durable_memory_correction')
    observation = value['manager_before']
    need(type(observation) is dict and observation.get('Id') == profile['unit']['name']
         and observation.get('MainPID') == '0' and observation.get('ActiveState') in ('failed', 'inactive')
         and h._number(value['recorded_at']) and value['recorded_at'] <= time.time(), 'recovery_admin_main_absence')
    current_boot = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    need(h._definitely_dead_identity(old_main, current_boot), 'recovery_old_main_still_alive')
    need(type(terminal.get('owner_birth')) is str and terminal['owner_birth'].isdecimal(), 'recovery_owner_birth_unknown')
    owner = dict(pid=terminal['owner_pid'], birth_ticks=int(terminal['owner_birth']), boot_id=old_main['boot_id'])
    need(h._integer(owner['pid'], 2) and h._integer(owner['birth_ticks'], 1)
         and h._definitely_dead_identity(owner, current_boot), 'recovery_old_owner_still_alive')
    _recovery_scheduler(request, value['scheduler_snapshot_ref'])
    return failed


def authorize_known_nonstarted_recovery(spec_ref, output):
    """Explicit admin-only metadata step while the actual unit has no MainPID.

    Source/config adoption precedes this call; unit ENV pins the returned digest
    afterward. Evidence snapshots are data under the SAME proof-root archive.
    This function never starts a service, allocates resources or removes a hold.
    """
    h = health_owner(); spec = h._document(spec_ref)
    h._keys(spec, RECOVERY_KEYS - {'schema', 'recorded_at', 'manager_before'}, 'recovery_authorization_spec')
    profile = h._profile(spec['profile_ref'])
    config_ref = next(pin for pin in profile['source_descriptors'] if pin['path'] in profile['unit']['argv']
        and pin['path'].endswith('.json') and h._document(pin).get('schema') == CONFIG_SCHEMA)
    config, _, _ = load_configuration(config_ref)
    root = Path(profile['proof_root']); output = h._path(str(output))
    need((root / 'reconciliation-v01') in output.parents, 'recovery_authorization_outside_archive')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_NOFOLLOW), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = h._manager(profile['unit']['name'])
        need(manager.get('MainPID') == '0' and manager.get('ActiveState') in ('failed', 'inactive'), 'recovery_admin_requires_no_main')
        value = dict(spec, schema=RECOVERY_SCHEMA, recorded_at=time.time(), manager_before=manager)
        _validate_nonstarted_recovery(value, spec['profile_ref'], profile, config)
        return put(output, value)


def reconcile_known_nonstarted(recovery_ref, profile_ref, profile, config, opening_identity):
    """One exact negative acknowledgement; no closed.json or health receipt."""
    h = health_owner(); h._within(recovery_ref, Path(profile['proof_root']) / 'reconciliation-v01')
    value = h._document(recovery_ref)
    need(os.environ.get(RECOVERY_SHA_ENV) == recovery_ref['sha256'], 'recovery_receipt_not_unit_pinned')
    argv = profile['unit']['argv']; flag = '--known-nonstarted-recovery'
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
         and argv[argv.index(flag) + 1] == recovery_ref['path'], 'recovery_receipt_not_in_unit_argv')
    failed = _validate_nonstarted_recovery(value, profile_ref, profile, config)
    ack_path = failed / 'nonstarted-reconciliation.json'
    ack = dict(schema='agent-supervisor/scientific-nonstarted-reconciliation@1',
        recovery_ref=recovery_ref, hold_ref=value['hold_ref'], failed_cycle_root=str(failed),
        consuming_main_identity=opening_identity, health_credit=False, completion_authority=False,
        dispatch_authority=False)
    numbers = []
    for cycle in Path(profile['proof_root']).glob('cycle-*'):
        intent = h._document(h.descriptor(cycle / 'intent.json'))
        number = intent.get('cycle_number', intent.get('sequence'))
        need(h._integer(number, 1, config['max_cycles']), 'recovery_cycle_counter_unknown')
        need(cycle.name.startswith('cycle-%08d-' % number), 'recovery_cycle_number_path')
        need(cycle == failed or (cycle / 'closed.json').is_file(), 'recovery_other_unfinished_cycle')
        numbers.append(number)
    need(sorted(numbers) == list(range(1, len(numbers) + 1)), 'recovery_cycle_counter_gap_or_duplicate')
    need(len(numbers) < config['max_cycles'], 'recovery_cycle_budget_exhausted')
    if os.path.lexists(ack_path):
        need(h._document(h.descriptor(ack_path)) == ack, 'recovery_authorization_already_consumed_by_other_main')
    else:
        put(ack_path, ack)
    return len(numbers)


def _closed_idle_source_inventory(value, profile):
    """Authenticate the adopted generation; historical owner/driver may differ."""
    h = health_owner(); source = h._document(value['source_commitment_ref'])
    source_root = Path(profile['owner_source']['path']).parents[3]
    need(source.get('schema') == 'legal-ir-scientific-source-commitment/v1'
         and source.get('runtime_activation_or_task_authority') is False
         and source.get('source_root') == str(source_root), 'closed_idle_source_commitment')
    inventory_ref = source['source_inventory']
    need(Path(profile['proof_root']).parent in Path(inventory_ref['path']).parents,
         'closed_idle_inventory_not_durable')
    inventory = h._document(inventory_ref); rows = inventory.get('files')
    need(inventory.get('schema') == 'legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(key) == source.get(key) for key in ('source_root', 'source_commit', 'source_tree'))
         and all(type(source.get(key)) is str and len(source[key]) == 40
                 and set(source[key]) <= set('0123456789abcdef') for key in ('source_commit', 'source_tree'))
         and type(rows) is list and 0 < len(rows) <= 4096 and inventory.get('file_count') == len(rows)
         and all(type(row) is dict and type(row.get('path')) is str for row in rows),
         'closed_idle_inventory_shape')
    indexed = {row['path']: row for row in rows}
    selected = [pin for pin in profile['source_descriptors']
                if Path(pin['path']).is_relative_to(source_root) and Path(pin['path']).suffix == '.py']
    need(len(indexed) == len(rows)
         and all(profile[field] in selected for field in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'))
         and all(all(indexed.get(pin['path'], {}).get(key) == pin[key]
                     for key in ('path', 'bytes', 'sha256')) for pin in selected),
         'closed_idle_inventory_current_source_mismatch')


def _closed_idle_origins(origins, prior, request, plan, negative):
    """Historical incomplete execution provenance remains unlabelled; no health credit."""
    h = health_owner()
    need(type(origins) is dict and origins.get('schema') == 'legal-ir-pre-execution-project-import-origins/v1'
         and origins.get('project_origins_checked_before_module_execution') is True
         and origins.get('denials') == [], 'closed_idle_origin_guard_unknown')
    rows = origins.get('project_imports'); pins = {pin['path']: pin for pin in prior['source_descriptors']}
    need(type(rows) is list and 1 <= len(rows) <= 4096
         and len({row.get('module') for row in rows if type(row) is dict}) == len(rows),
         'closed_idle_origin_population')
    missing = []
    for row in rows:
        need(type(row) is dict and h._text(row.get('module')), 'closed_idle_origin_shape')
        pin = {key: row.get(key) for key in ('path', 'bytes', 'sha256')}
        need(pins.get(row.get('path')) == pin and request['source_sha256'].get(pin['path']) == pin['sha256'],
             'closed_idle_origin_unpinned')
        h._bound_project_source(pin, prior)
        if 'execution_profile' not in row:
            missing.append(row)
        else:
            need(row['execution_profile'] == 'compiled_exact_authenticated_source_bytes',
                 'closed_idle_origin_other_execution_state')
    need(len(missing) == 1 and missing[0].get('module') == 'ipfs_datasets_py.ipfs_backend_router'
         and missing[0] == negative.get('exact_unlabeled_origin')
         and negative.get('actual_origin_rows') == len(rows)
         and negative.get('compiled_authenticated_origin_rows') == len(rows) - 1,
         'closed_idle_not_exact_unlabelled_origin_refusal')
    command = plan['native_metadata_cli']; declared = command['project_import_plan']['source_pins']
    need(type(declared) is list and missing[0]['path'] == str(Path(command['project_import_plan']['datasets_root']) /
         'ipfs_datasets_py/ipfs_backend_router.py') and pins[missing[0]['path']] in declared,
         'closed_idle_resolution_source_binding')
    # Do not add execution_profile, rewrite origins, or call a health publisher.
    return rows


def _closed_idle_observation(result, request, plan, origins, prior, native, terminal):
    """Validate the read-only outcome separately from rejected health evidence."""
    h = health_owner(); command = plan.get('native_metadata_cli', {}); binding = command.get('observation_binding', {})
    h._keys(result, {'schema', 'backend', 'database_path', 'task_scope', 'observer_source', 'observer_callable',
        'catalogue_binding', 'ready_task_count', 'selection_idle_reason', 'unchanged', 'write_count',
        'provider_dispatched', 'attempt_consumed', 'completion_authority'}, 'closed_idle_result_shape')
    need(plan.get('schema') == 'native-embedded-scoped-observation-plan/v1'
         and plan.get('operation') == 'native_embedded_scoped_observation_v1'
         and result['schema'] == 'native-embedded-daemon-observation@1' and result['backend'] == 'embedded'
         and result['observer_callable'] == 'observe_embedded_database_without_dispatch'
         and result['selection_idle_reason'] == 'no_scoped_ready_tasks'
         and result['task_scope'] == plan.get('task_scope') == prior['task_scope']
         and type(result['ready_task_count']) is int and result['ready_task_count'] == 0
         and type(result['write_count']) is int and result['write_count'] == 0
         and result['unchanged'] is True and all(result[key] is False for key in
             ('provider_dispatched', 'attempt_consumed', 'completion_authority')),
         'closed_idle_outcome_not_readonly')
    db = request['database_preimage']
    h._keys(db, {'path', 'bytes', 'sha256'}, 'closed_idle_database_descriptor')
    need(h._integer(db['bytes'], 1, 128 * 1024**2) and type(db['sha256']) is str
         and len(db['sha256']) == 64 and set(db['sha256']) <= set('0123456789abcdef'), 'closed_idle_database_descriptor_bounds')
    h._path(db['path'])
    need(terminal.get('database_preimage_before_grants') == db == terminal.get('database_preimage_observed')
         == terminal.get('database_postimage') == binding.get('database_preimage')
         and result['database_path'] == db['path'] and h.database_descriptor(db['path']) == db,
         'closed_idle_database_history_or_current_changed')
    source = result['observer_source']; route = command.get('import_head_capsule_Git', {}).get('route_source')
    need(command.get('schema') == 'native-embedded-scoped-observation-cli/v2'
         and command.get('runtime_root') == plan.get('accelerate_root')
         and binding.get('schema') == 'native-embedded-daemon-observation-binding@1'
         and source == binding.get('observer_source') and source in prior['source_descriptors']
         and route in prior['source_descriptors']
         and all(request['source_sha256'].get(pin['path']) == command['source_sha256'].get(pin['path']) == pin['sha256']
                 for pin in (source, route)), 'closed_idle_observer_or_route_source')
    rows = origins['project_imports']
    for name, pin in (('ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap', source),
                      ('ipfs_accelerate_py.agent_implementation_route', route)):
        need(any(row.get('module') == name and all(row.get(key) == pin[key] for key in ('path', 'bytes', 'sha256'))
                 and row.get('execution_profile') == 'compiled_exact_authenticated_source_bytes' for row in rows),
             'closed_idle_observer_body_not_compiled')
    catalogue = result['catalogue_binding']
    h._keys(catalogue, {'database_preimage', 'plan_root_cid', 'repository_tree_id', 'source_schema',
        'event_cursor', 'task_statuses'}, 'closed_idle_catalogue_shape')
    need(catalogue['database_preimage'] == db and binding.get('task_scope') == prior['task_scope']
         and catalogue['source_schema'] == binding.get('source_schema') ==
             'ipfs_accelerate_py/agent-supervisor/database-task-source@1'
         and all(catalogue[key] == binding.get(key) == prior['catalogue'][key]
                 for key in ('plan_root_cid', 'repository_tree_id'))
         and h._integer(catalogue['event_cursor']) and catalogue['event_cursor'] == native['event_cursor'], 'closed_idle_catalogue_binding')
    expected = {row['task_cid']: {key: row[key] for key in ('task_cid', 'status', 'revision')}
                for row in native['tasks'] if row['task_cid'] in prior['task_scope']}
    tasks = catalogue['task_statuses']
    need(type(tasks) is list and len(tasks) == len(expected) == len(prior['task_scope'])
         and len({row.get('task_cid') for row in tasks if type(row) is dict}) == len(tasks)
         and all(type(row) is dict and set(row) == {'task_cid', 'status', 'revision'}
                 and h._integer(row.get('revision'), 1)
                 and row == expected.get(row['task_cid']) and row['status'] in QUIET_STATUSES for row in tasks),
         'closed_idle_catalogue_tasks_changed')


def _validate_closed_idle_recovery(value, profile_ref, profile, config):
    """Explicit negative reconciliation of exactly two spent idle observations."""
    h = health_owner(); h._keys(value, CLOSED_IDLE_KEYS, 'closed_idle_recovery_shape')
    need(value['schema'] == CLOSED_IDLE_SCHEMA and value['recovery_kind'] == 'known_closed_idle_health_rejection'
         and value['authorization'] == 'resume_genuine_scoped_idle' and value['profile_ref'] == profile_ref
         and h._profile(profile_ref) == profile and value['max_cycles'] == config['max_cycles']
         and config['max_cycles'] > 2 and config['execution_templates'] == [], 'closed_idle_recovery_authority')
    root = Path(profile['proof_root']); archive = root / 'reconciliation-v02'; total = 0
    for key in CLOSED_IDLE_ARCHIVE_FIELDS:
        h._within(value[key], archive); h._bound(value[key]); total += value[key]['bytes']
    need(total <= 1024**2, 'closed_idle_archive_bound')
    prior = h._document(value['prior_profile_snapshot_ref'])
    need(_same_payload(value['prior_profile_ref'], value['prior_profile_snapshot_ref'])
         and value['prior_profile_ref']['path'] == profile_ref['path'], 'closed_idle_prior_profile_snapshot')
    for key in ('catalogue', 'task_scope', 'proof_root', 'receipt_path', 'max_age_seconds', 'max_cycle_seconds', 'cycle_contract'):
        need(prior.get(key) == profile[key], 'closed_idle_prior_scope_changed')
    need(all(prior['unit'][key] == profile['unit'][key] for key in
         ('name', 'interpreter', 'working_directory', 'environment')), 'closed_idle_prior_unit_changed')
    _closed_idle_source_inventory(value, profile)
    old_auth = h._document(value['prior_authorization_ref']); old_ack = h._document(value['prior_ack_ref'])
    capture = h._document(value['prior_administrative_capture_ref']); negative = h._document(value['negative_audit_ref'])
    need(old_auth.get('schema') == RECOVERY_SCHEMA and old_auth.get('profile_ref') == value['prior_profile_ref']
         and old_auth.get('authorization') == 'resume_genuine_scoped_idle'
         and old_auth.get('max_cycles') == config['max_cycles']
         and old_ack.get('schema') == 'agent-supervisor/scientific-nonstarted-reconciliation@1'
         and old_ack.get('recovery_ref') == value['prior_authorization_ref']
         and all(old_ack.get(key) is False for key in ('health_credit', 'completion_authority', 'dispatch_authority')),
         'closed_idle_prior_authorization_or_ack')
    original_plan = h._document(old_auth['activation_plan_ref'])
    need(all(config[key] == original_plan['main_loop_limits'][key] for key in
             ('max_cycles', 'poll_seconds', 'max_proof_bytes')), 'closed_idle_original_budget_changed')
    first_request = h._document(old_auth['request_ref'])
    first_terminal = original_terminal(h._bound(old_auth['owner_stdout_ref']), old_auth['request_ref'])
    first_negative = h._document(old_auth['negative_audit_ref'])
    need(first_terminal.get('completed') is False and first_terminal.get('child_launch_attempted') is False
         and first_terminal.get('child_reaped') is False and first_terminal.get('child_returncode') is None
         and first_terminal.get('control_effects') == 'not_started' and first_terminal.get('storage_status') == 'not_requested'
         and first_terminal.get('source_READ_closed') is True and first_terminal.get('original_native_compute_closed') is True
         and first_terminal.get('absolute_resource_operation_deadline_closed') is True
         and first_terminal.get('cleanup_errors') == [] and first_terminal.get('received_signals') == []
         and first_terminal.get('task_completion_authority_granted_by_resource_owner') is False
         and first_terminal.get('database_preimage_before_grants') == first_request.get('database_preimage')
         and first_negative.get('negative_terminal_and_hold_verified') is True
         and first_negative.get('original_terminal') == first_terminal
         and first_negative.get('persistent_hold') == h._document(value['global_hold_ref']),
         'closed_idle_first_nonstarted_history_changed')
    first_capture = h._document(old_auth['activation_capture_ref'])
    first_main = first_capture['opening_actual_main_identity']
    first_prior = h._document(old_auth['prior_profile_snapshot_ref'])
    h._stored_identity(first_main, old_auth['prior_profile_ref'], first_prior)
    old_main = capture.get('opening_identity'); h._stored_identity(old_main, value['prior_profile_ref'], prior)
    need(capture.get('schema') == 'scientific-explicit-known-nonstarted-recovery-actual-capture/v1'
         and capture.get('accepted') is False and capture.get('start_requested') is True
         and capture.get('source_leases_closed_before_start') is True and capture.get('hold_removed') is False
         and capture.get('native_or_task_or_epoch_or_policy_reset') is False
         and capture.get('recovery_authorization_ref') == value['prior_authorization_ref']
         and old_ack.get('consuming_main_identity') == old_main, 'closed_idle_original_main_lineage')
    need(negative.get('schema') == 'scientific-native-completed-idle-health-failure-independent-negative-audit/v1'
         and _same_payload(negative.get('actual_administrative_capture'), value['prior_administrative_capture_ref'])
         and negative.get('actual_profile_ref') == value['prior_profile_ref']
         and negative.get('authorization_ref') == value['prior_authorization_ref']
         and negative.get('negative_ack_ref') == value['prior_ack_ref']
         and negative.get('accepted_as_executor_health_or_activation') is False
         and negative.get('completed_native_read_only_cycle_independently_verified') is True
         and negative.get('native_source_process_storage_closed') is True
         and negative.get('task_or_provider_or_completion_effects') is False
         and negative.get('database_unchanged_during_original_cycle') is True
         and negative.get('exact_original_cycle_number') == 2 and negative.get('health_sequence_attempted') == 1,
         'closed_idle_independent_negative_join')
    hold = h._document(value['global_hold_ref']); cycle_hold = h._document(value['cycle_hold_ref'])
    failed = Path(value['cycle_proof_ref']['path']).parent
    need(value['global_hold_ref'] == old_auth['hold_ref'] == old_ack['hold_ref'] == negative['global_original_hold_ref']
         and value['global_hold_ref']['path'] == str(root / 'executor-held.json')
         and hold.get('reason') == 'original_owner_failed' and hold.get('automatic_retry_admitted') is False
         and value['cycle_hold_ref']['path'] == str(failed / 'cycle-held.json')
         and value['cycle_hold_ref'] == negative['new_cycle_hold_ref']
         and cycle_hold.get('schema') == 'agent-supervisor/scientific-cycle-held@1'
         and cycle_hold.get('reason') == 'original_import_origin_unbound'
         and cycle_hold.get('automatic_retry_admitted') is False, 'closed_idle_exact_two_holds')
    cycles = sorted(root.glob('cycle-*'))
    need(len(cycles) == 2 and failed.parent == root and failed.name.startswith('cycle-00000002-')
         and cycles[0] == Path(old_auth['failed_cycle_root'])
         and cycles[1] == failed and value['prior_ack_ref']['path'] == str(cycles[0] / 'nonstarted-reconciliation.json')
         and all(not (cycle / 'closed.json').exists() for cycle in cycles)
         and not os.path.lexists(profile['receipt_path']), 'closed_idle_cycle_history_or_receipt')
    intent = h._document(h.descriptor(failed / 'intent.json')); proof = h._document(value['cycle_proof_ref'])
    need(intent.get('kind') == 'genuine_scoped_idle' and intent.get('automatic_retry_admitted') is False
         and type(intent.get('cycle_number')) is int and intent['cycle_number'] == 2
         and type(intent.get('sequence')) is int and intent['sequence'] == 1
         and intent.get('request_ref') == proof.get('request_ref')
         and proof.get('kind') == 'genuine_scoped_idle' and proof.get('sequence') == 1,
         'closed_idle_original_intent')
    for key in ('request_ref', 'execution_request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref'):
        need(proof[key] == negative.get(key), 'closed_idle_negative_artifact_join')
    request, terminal, result, plan, origins, native = (h._document(proof[key]) for key in
         ('request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref'))
    need(original_terminal(h._raw(failed / 'owner.stdout'), proof['request_ref']) == terminal,
         'closed_idle_original_stdout_terminal')
    native_identity = h._native(native, prior, now=proof['finished_at'])
    need(proof['started_at'] <= native['checked_at_epoch'] <= proof['finished_at'], 'closed_idle_native_capture_outside_cycle')
    execution = h._document(proof['execution_request_ref'])
    need(execution.get('completion_authority') is False and execution.get('provider_authority') is False,
         'closed_idle_execution_authority')
    # Reuse every original terminal/process/native/storage check preceding the
    # historical label refusal. This is an expected rejection, never health.
    try:
        h._cycle(value['cycle_proof_ref'], prior, old_main, native_identity,
                 now=proof['finished_at'], monotonic_ns=proof['finished_monotonic_ns'])
    except h.ExecutorHealthError as exc:
        need(str(exc) == 'original_import_origin_unbound', 'closed_idle_other_original_cycle_refusal')
    else:
        raise ExecutorHold('closed_idle_historical_health_was_not_rejected')
    _closed_idle_origins(origins, prior, request, plan, negative)
    _closed_idle_observation(result, request, plan, origins, prior, native, terminal)
    need(terminal.get('absolute_resource_operation_deadline_closed') is True
         and terminal.get('child_launch_attempted') is True
         and terminal.get('control_effects') == 'actor_exit_zero_independent_task_receipt_still_required'
         and terminal.get('task_completion_authority_granted_by_resource_owner') is False,
         'closed_idle_original_terminal_authority')
    current = h._document(value['current_native_ref']); h._native(current, profile)
    def scoped_tasks(report):
        return sorted(({key: row[key] for key in ('task_cid', 'status', 'revision')}
                       for row in report['tasks'] if row['task_cid'] in profile['task_scope']), key=lambda row: row['task_cid'])
    need(current['event_cursor'] == native['event_cursor'] and scoped_tasks(current) == scoped_tasks(native),
         'closed_idle_current_tasks_changed')
    boot = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    need(h._definitely_dead_identity(first_main, boot), 'closed_idle_first_main_alive')
    need(type(first_terminal.get('owner_birth')) is str and first_terminal['owner_birth'].isdecimal(),
         'closed_idle_first_owner_birth_unknown')
    first_owner = dict(pid=first_terminal['owner_pid'], birth_ticks=int(first_terminal['owner_birth']), boot_id=first_main['boot_id'])
    need(h._integer(first_owner['pid'], 2) and h._integer(first_owner['birth_ticks'], 1)
         and h._definitely_dead_identity(first_owner, boot), 'closed_idle_first_owner_alive')
    need(h._definitely_dead_identity(old_main, boot), 'closed_idle_prior_main_alive')
    for old in terminal['sampled_process_custody_final']['observed_identities']:
        need(type(old) is dict and h._integer(old.get('pid'), 2) and h._integer(old.get('birth_ticks'), 1)
             and h._definitely_dead_identity(dict(old, boot_id=old_main['boot_id']), boot), 'closed_idle_prior_descendant_alive')
    owner = proof['owner_observation']
    need(h._definitely_dead_identity(owner, boot), 'closed_idle_prior_owner_alive')
    manager = value['manager_before']
    need(manager.get('Id') == profile['unit']['name'] and manager.get('MainPID') == '0'
         and manager.get('ActiveState') in ('failed', 'inactive')
         and h._number(value['recorded_at']) and value['recorded_at'] <= time.time(), 'closed_idle_admin_main_absence')
    _recovery_scheduler(request, value['scheduler_snapshot_ref'])
    _recovery_scheduler(h._document(old_auth['request_ref']), value['scheduler_snapshot_ref'])
    # The replacement source must bind the corrected guard plus new owners;
    # historical role hashes are evidence and are never imposed on the new run.
    future = h._document(h._document(config['idle_template_ref'])['plan_ref'])['native_metadata_cli']
    guard = future.get('pre_body_guard'); prior_guard = plan['native_metadata_cli'].get('pre_body_guard')
    need(type(guard) is dict and guard in profile['source_descriptors'] and guard != prior_guard
         and all(profile[key]['sha256'] != prior[key]['sha256'] for key in ('owner_source', 'driver_source')),
         'closed_idle_cause_source_not_adopted')
    h._bound(guard)
    return failed


def authorize_known_closed_idle_recovery(spec_ref, output):
    """Existing administrative protocol, distinct typed one-use authorization."""
    h = health_owner(); spec = h._document(spec_ref)
    h._keys(spec, CLOSED_IDLE_KEYS - {'schema', 'recorded_at', 'manager_before'}, 'closed_idle_authorization_spec')
    profile = h._profile(spec['profile_ref'])
    config_ref = next(pin for pin in profile['source_descriptors'] if pin['path'] in profile['unit']['argv']
        and pin['path'].endswith('.json') and h._document(pin).get('schema') == CONFIG_SCHEMA)
    config, _, _ = load_configuration(config_ref); root = Path(profile['proof_root']); output = h._path(str(output))
    need(output.parent == root / 'reconciliation-v02' and output.name == 'known-closed-idle-authorization.json',
         'closed_idle_authorization_versioned_path')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_NOFOLLOW), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = h._manager(profile['unit']['name'])
        need(manager.get('MainPID') == '0' and manager.get('ActiveState') in ('failed', 'inactive'),
             'closed_idle_admin_requires_no_main')
        value = dict(spec, schema=CLOSED_IDLE_SCHEMA, recorded_at=time.time(), manager_before=manager)
        _validate_closed_idle_recovery(value, spec['profile_ref'], profile, config)
        return put(output, value)


def reconcile_known_closed_idle(recovery_ref, profile_ref, profile, config, opening_identity):
    """Retain both holds and consumed old AUTH/ACK; spend neither a third run nor health."""
    h = health_owner(); root = Path(profile['proof_root'])
    need(recovery_ref['path'] == str(root / 'reconciliation-v02/known-closed-idle-authorization.json'),
         'closed_idle_recovery_versioned_path')
    value = h._document(recovery_ref)
    need(os.environ.get(CLOSED_IDLE_SHA_ENV) == recovery_ref['sha256'], 'closed_idle_receipt_not_unit_pinned')
    argv = profile['unit']['argv']; flag = '--known-closed-idle-recovery'
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv) and argv[argv.index(flag) + 1] == recovery_ref['path']
         and '--known-nonstarted-recovery' not in argv, 'closed_idle_receipt_not_in_unit_argv')
    failed = _validate_closed_idle_recovery(value, profile_ref, profile, config)
    ack_path = failed / 'closed-idle-reconciliation-v01.json'
    ack = dict(schema='agent-supervisor/scientific-closed-idle-reconciliation@1', recovery_ref=recovery_ref,
        retained_global_hold_ref=value['global_hold_ref'], retained_cycle_hold_ref=value['cycle_hold_ref'],
        prior_authorization_ref=value['prior_authorization_ref'], prior_ack_ref=value['prior_ack_ref'],
        rejected_cycle_proof_ref=value['cycle_proof_ref'], consuming_main_identity=opening_identity,
        spent_cycle_count=2, next_cycle_number=3, first_health_sequence=1,
        health_credit=False, completion_authority=False, dispatch_authority=False, historical_origins_relabelled=False)
    if os.path.lexists(ack_path):
        need(h._document(h.descriptor(ack_path)) == ack, 'closed_idle_authorization_consumed_by_other_main')
    else:
        put(ack_path, ack)
    return 2


def _recovery_failure(root, cycle_number, sequence, reason):
    """Retain a later failure even while the original global hold is immutable."""
    matches = list(root.glob('cycle-%08d-*' % cycle_number))
    need(len(matches) <= 1, 'recovery_duplicate_cycle_number')
    if matches:
        cycle = matches[0]
    else:
        cycle = root / ('cycle-%08d-%s' % (cycle_number, uuid.uuid4().hex)); cycle.mkdir(mode=0o700)
        put(cycle / 'intent.json', dict(kind='not_selected', sequence=sequence,
            cycle_number=cycle_number, automatic_retry_admitted=False))
    if not (cycle / 'cycle-held.json').exists():
        put(cycle / 'cycle-held.json', dict(schema='agent-supervisor/scientific-cycle-held@1',
            reason=str(reason)[:2048], recorded_at=time.time(), automatic_retry_admitted=False))


def require_main_process(profile_ref, profile):
    """Authenticate the producer before spawning any original owner."""
    h = health_owner()
    need(threading.current_thread() is threading.main_thread(), 'service_not_main_thread')
    need(Path(__file__).resolve() == Path(profile['main_entrypoint']['path']) and
        Path(h.__file__).resolve() == Path(profile['health_module']['path']), 'service_live_source_origin_changed')
    identity = h._identity(profile_ref, profile, h._manager, h._process)
    need(identity['pid'] == os.getpid(), 'service_not_systemd_main_pid')
    return identity


def run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop, *, cycle_number=None, catalogue_parser=None):
    h = health_owner()
    need(cycle_number is None or h._integer(cycle_number, 1), 'cycle_number')
    root = Path(profile['proof_root']) / ('cycle-%08d-%s' % (sequence if cycle_number is None else cycle_number, uuid.uuid4().hex))
    template=health_owner()._document(template_ref)
    selected_plan=health_owner()._document(template['plan_ref'])
    if selected_plan.get('native_metadata_cli',{}).get('schema')=='native-embedded-scoped-observation-cli/v3':
        original_request=health_owner()._document(template['request_ref'])
        need(original_request.get('storage_bytes')==384*1024**2 and type(original_request.get('storage_bytes'))is int,'reuse_original_storage_grant_changed')
        check_proof_budget(Path(profile['proof_root']),config['max_proof_bytes'],
            reserve_bytes=REUSE_CYCLE_RESERVE_BYTES,reserve_files=REUSE_CYCLE_RESERVE_FILES)
    root.mkdir(mode=0o700)
    process = None
    try:
        request_ref, plan_ref, links = prepare_cycle(template_ref, kind, config, profile, root)
        intent = {'kind': kind, 'sequence': sequence, 'request_ref': request_ref, 'automatic_retry_admitted': False}
        if cycle_number is not None: intent['cycle_number'] = cycle_number
        put(root / 'intent.json', intent)
        stdout_path, stderr_path = root / 'owner.stdout', root / 'owner.stderr'
        with os.fdopen(os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as out, \
             os.fdopen(os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as err:
            argv = [profile['unit']['argv'][0], '-B', profile['owner_source']['path'],
                '--request', request_ref['path'], '--expected-request-sha256', request_ref['sha256']]
            process = subprocess.Popen(argv, cwd=profile['unit']['working_directory'],
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True)
            handle = h.track_original_cycle(profile_ref, process, request_ref, sequence=sequence)
            deadline = time.monotonic() + profile['max_cycle_seconds']
            while process.poll() is None:
                need(not stop.is_set() and time.monotonic() < deadline, 'original_cycle_interrupted_or_timeout')
                need(stdout_path.stat().st_size <= h.MAX_JSON and stderr_path.stat().st_size <= h.MAX_JSON,
                    'original_owner_output_bound')
                stop.wait(0.1)
            original_returncode=process.wait()
            if original_returncode!=0 and kind=='genuine_scoped_idle' and not stop.is_set():
                try:
                    request=h._document(request_ref)
                    witness=source_admission_deferral(profile,request_ref,request,root,kind,process,h.cycle_start_fields(handle),original_returncode)
                except Exception:
                    # An unproved optional deferral preserves the original hold.
                    pass
                else:
                    raise SourceAdmissionDeferred(witness['path'])
            need(original_returncode == 0, 'original_owner_failed')
        terminal_ref = put(root / 'terminal.json', original_terminal(h._raw(stdout_path), request_ref))
        native = (observe_native(config) if catalogue_parser is None else
                  observe_native(config,catalogue_parser=catalogue_parser))
        native_ref = put(root / 'native-diagnostic.json', native)
        proof = dict(h.cycle_start_fields(handle), schema=h.CYCLE_SCHEMA, kind=kind,
            finished_at=time.time(), finished_monotonic_ns=time.monotonic_ns(),
            request_ref=request_ref, plan_ref=plan_ref, terminal_ref=terminal_ref,
            execution_request_ref=h.descriptor(links['execution_request']),
            result_ref=h.descriptor(links['result']), origins_ref=h.descriptor(links['origins']),
            native_diagnostic_ref=native_ref)
        proof_ref = put(root / 'cycle-proof.json', proof)
        try:
            receipt_ref = h.publish_after_native_cycle(profile_ref, proof_ref, native,
                sequence=sequence, cycle_handle=handle)
        except Exception as error:
            if (kind=='genuine_scoped_idle' and not stop.is_set()
                    and type(error)is getattr(h,'ExecutorHealthError',None)
                    and str(error)in CATALOGUE_RACE_REJECTIONS):
                try:
                    witness=catalogue_race_deferral(profile_ref,profile,proof_ref,native,kind,process,stop,cycle_handle=handle)
                except Exception:
                    # Optional negative recognition never replaces the original
                    # publication failure when closure or source is unknown.
                    pass
                else:
                    raise CatalogueRaceDeferred(witness['path'])
            raise
        put(root / 'closed.json', {'cycle_proof_ref': proof_ref, 'receipt_ref': receipt_ref})
        return receipt_ref
    except (SourceAdmissionDeferred,CatalogueRaceDeferred):
        raise
    except BaseException as exc:
        # A signal is handed to the original owner, which owns cancellation and
        # custody. No synthetic closure, SIGKILL, reservation release or replay.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        persistent_hold(profile['proof_root'], root, str(exc))
        if cycle_number is not None: _recovery_failure(Path(profile['proof_root']), cycle_number, sequence, exc)
        raise


def _generation_census(reference):
    """Verify every retained byte/path; never remove, relocate or relabel it."""
    h=health_owner(); value=h._document(reference)
    h._keys(value,{'schema','root','manifests','file_count','source_bytes','directory_sha256'},'generation_census_shape')
    need(value['schema']=='agent-supervisor/scientific-proof-generation-census@1','generation_census_schema')
    root=h._path(value['root']); info=root.lstat()
    need(any(parent in root.parents for parent in ORIGINAL_NAMED_ROOTS)
         and stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid() and not info.st_mode&0o022,
         'retained_generation_directory_identity')
    need(not Path(reference['path']).lstat().st_mode&0o222,'generation_census_not_immutable')
    manifests=value['manifests']; need(type(manifests)is list and 1<=len(manifests)<=64,'generation_manifest_population')
    expected={}; total=0
    for reference in manifests:
        node=h._document(reference)
        need(not Path(reference['path']).lstat().st_mode&0o222,'generation_file_manifest_not_immutable')
        h._keys(node,{'schema','root','files'},'generation_file_manifest_shape')
        need(node['schema']=='agent-supervisor/scientific-proof-file-manifest@1' and node['root']==str(root)
             and type(node['files'])is list and 0<len(node['files'])<=4096,'generation_file_manifest_population')
        for row in node['files']:
            h._keys(row,{'relative_path','bytes','sha256','mode'},'generation_retained_file_shape')
            rel=row['relative_path']; need(type(rel)is str and rel and not Path(rel).is_absolute()
                and '..'not in Path(rel).parts and str(Path(rel))==rel and rel not in expected,
                'generation_retained_file_alias')
            path=root/rel; pin={key:row[key] for key in ('bytes','sha256')};pin['path']=str(path)
            h._bound(pin,128*1024**2)
            need(type(row['mode'])is int and 0<=row['mode']<=0o777 and not row['mode']&0o022
                 and stat.S_IMODE(path.lstat().st_mode)==row['mode'],'generation_retained_file_mode')
            expected[rel]=pin;total+=row['bytes']
            need(len(expected)<=22337 and total<=2*1024**3,'generation_retained_population_bound')
    observed=set();directories=[('.',stat.S_IMODE(root.lstat().st_mode))]
    for path in root.rglob('*'):
        info=path.lstat();need(not stat.S_ISLNK(info.st_mode),'generation_retained_symlink')
        if stat.S_ISDIR(info.st_mode):
            need(info.st_uid==os.geteuid() and not info.st_mode&0o022,'generation_retained_directory_identity')
            directories.append((path.relative_to(root).as_posix(),stat.S_IMODE(info.st_mode)))
            continue
        need(stat.S_ISREG(info.st_mode) and info.st_uid==os.geteuid() and info.st_nlink==1,'generation_retained_nonordinary')
        observed.add(path.relative_to(root).as_posix())
    directory_digest=hashlib.sha256(json.dumps(sorted(directories),separators=(',',':')).encode()).hexdigest()
    need(value['directory_sha256']==directory_digest,'generation_retained_directory_modes_changed')
    need(observed==set(expected) and type(value['file_count'])is int and value['file_count']==len(expected)
         and type(value['source_bytes'])is int and value['source_bytes']==total,'generation_census_changed')
    return value


def _released_initialization_record(request,terminal):
    """Read the original ledger; never acquire, renew, release or normalize it."""
    h=health_owner();path=Path(request['resource_ledger'])
    need(path==ORIGINAL_RESOURCE_LEDGER,'generation_original_ledger_path_changed')
    ref=h.database_descriptor(path);raw=path.read_bytes()
    need(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
         and h.database_descriptor(path)==ref,'generation_original_ledger_changed')
    value=h._decode(raw)
    need(value.get('schema')=='daemon-resource-reservations-v1' and value.get('limit_bytes')==GLOBAL_RETAINED_BYTES
         and [row['path'] for row in value.get('roots',[])]==[str(p) for p in ORIGINAL_NAMED_ROOTS],
         'generation_original_ledger_policy_changed')
    for row in value['roots']:
        info=Path(row['path']).stat()
        need((info.st_dev,info.st_ino)==(row['device'],row['inode']),'generation_original_root_identity_changed')
    reservation=terminal['reservation'];record=value.get('reservations',{}).get(reservation['reservation_id'])
    need(type(record)is dict and record.get('status')=='released' and record.get('artifacts_durable_asserted')is True
         and record.get('owner_pid')==terminal['owner_pid']==reservation['record']['owner_pid']
         and record.get('native_lease_id')==reservation['resource_lease']['lease_id']
         and record.get('scheduler_policy_sha256')==request['expected_scheduler_policy_sha256']
         and record.get('scheduler_state_path')==request['scheduler_state_path']
         and record.get('workload')=='canonical_parented_phase' and record.get('scheduler_lane')=='orchestration'
         and record.get('attempt_directory',{}).get('path')==request['attempt_directory']
         and all(record.get(key)==request[key] for key in ('cpu_slots','memory_mb','child_process_slots','storage_bytes')),
         'generation_cache_original_reservation_not_released')
    return sum(row['storage_bytes'] for row in value['reservations'].values() if row.get('status')!='released')


def _validate_generation_handover(value,profile_ref,profile,config):
    h=health_owner();h._keys(value,GENERATION_KEYS,'generation_handover_shape')
    need(value['schema']==GENERATION_SCHEMA and value['authorization']=='one_bounded_source_generation_handover'
         and value['profile_ref']==profile_ref and h._profile(profile_ref)==profile,'generation_handover_authority')
    need(type(value['max_generations'])is int and value['max_generations']==2
         and type(value['retained_generations'])is list and len(value['retained_generations'])==1
         and type(value['generation_number'])is int and value['generation_number']==2 and type(value['global_max_bytes'])is int
         and value['global_max_bytes']==GLOBAL_RETAINED_BYTES,'generation_finite_population_or_global_budget')
    prior=h._document(value['prior_profile_snapshot_ref']);oldconfig=h._document(value['prior_config_snapshot_ref'])
    need(_same_payload(value['prior_profile_snapshot_ref'],value['prior_profile_ref'])
         and _same_payload(value['prior_config_snapshot_ref'],value['prior_config_ref']), 'generation_prior_snapshot')
    for key in ('catalogue','task_scope','max_age_seconds','max_cycle_seconds','cycle_contract'):
        need(prior.get(key)==profile[key],'generation_prior_scope')
    need(oldconfig['execution_templates']==config['execution_templates']==[]
         and all(oldconfig[k]==config[k] for k in ('catalogue','max_cycles','max_proof_bytes','poll_seconds')),
         'generation_original_limits_or_authority_changed')
    need(prior['unit']['name']==profile['unit']['name']
         and prior['unit']['working_directory']==profile['unit']['working_directory'], 'generation_unit_scope')
    oldroot=Path(prior['proof_root']);root=Path(profile['proof_root'])
    need(root!=oldroot and root.parent==oldroot.parent and root not in oldroot.parents and oldroot not in root.parents
         and any(parent in root.parents for parent in ORIGINAL_NAMED_ROOTS),'generation_named_root_scope')
    censuses=[];seen=set()
    for ref in value['retained_generations']:
        census=_generation_census(ref)
        need(census['root']not in seen and Path(census['root']).parent==oldroot.parent,'generation_registry_duplicate_or_crossed_root')
        seen.add(census['root']);censuses.append(census)
    need(value['retained_census_ref'] in value['retained_generations']
         and h._document(value['retained_census_ref'])['root']==str(oldroot),'generation_prior_census_missing')
    need(all(row['source_bytes']<=oldconfig['max_proof_bytes'] for row in censuses),'generation_prior_byte_quota_changed')
    need(sum(row['source_bytes'] for row in censuses)+check_proof_budget(root,config['max_proof_bytes'])['bytes']
         +REUSE_CYCLE_RESERVE_BYTES<=value['global_max_bytes'],'generation_aggregate_storage_bound')
    cycles=sorted(oldroot.glob('cycle-*'));spent=value['spent_cycle_count']
    need(h._integer(spent,1,config['max_cycles']-1) and len(cycles)==spent,'generation_spent_history_changed')
    numbers=[]
    for cycle in cycles:
        intent=h._document(h.descriptor(cycle/'intent.json'));number=intent.get('cycle_number',intent.get('sequence'))
        need(h._integer(number,1,spent) and cycle.name.startswith('cycle-%08d-'%number),'generation_spent_number_changed')
        numbers.append(number)
        if number<spent:
            need((cycle/'closed.json').is_file() or (cycle/'nonstarted-reconciliation.json').is_file()
                 or (cycle/'closed-idle-reconciliation-v01.json').is_file(),'generation_other_unfinished_cycle')
    need(sorted(numbers)==list(range(1,spent+1)),'generation_spent_gap_or_duplicate')
    failed=Path(value['prior_cycle_hold_ref']['path']).parent
    need(failed==cycles[-1] and value['prior_intent_ref']['path']==str(failed/'intent.json')
         and not any((failed/name).exists() for name in ('request.json','owner.stdout','terminal.json','cycle-proof.json','closed.json')),
         'generation_quota_failure_not_before_owner')
    hold=h._document(value['prior_cycle_hold_ref']);intent=h._document(value['prior_intent_ref'])
    need(hold.get('schema')=='agent-supervisor/scientific-cycle-held@1' and hold.get('reason')=='proof_storage_bound'
         and hold.get('automatic_retry_admitted')is False and intent.get('cycle_number')==spent
         and intent.get('kind')=='not_selected' and intent.get('automatic_retry_admitted')is False,'generation_quota_hold_changed')
    need(value['prior_global_hold_ref']['path']==str(oldroot/'executor-held.json'),'generation_original_global_hold_path')
    h._document(value['prior_global_hold_ref'])
    proof_ref=value['prior_last_closed_cycle_ref'];proof=h._document(proof_ref);cycle=Path(proof_ref['path']).parent
    need(len(cycles)>1 and cycle==cycles[-2] and proof_ref['path']==str(cycle/'cycle-proof.json')
         and proof.get('kind')=='genuine_scoped_idle' and (cycle/'closed.json').is_file()
         and h._document(h.descriptor(cycle/'closed.json'))['cycle_proof_ref']==proof_ref,'generation_last_closed_proof')
    last_intent=h._document(h.descriptor(cycle/'intent.json'))
    need(last_intent.get('kind')==proof['kind'] and last_intent.get('sequence')==proof['sequence']
         and last_intent.get('request_ref')==proof['request_ref'],'generation_latest_intent_proof_join')
    old_identity=value['prior_main_identity']
    h._stored_identity(old_identity,value['prior_profile_ref'],prior)
    native=h._document(proof['native_diagnostic_ref'])
    h._cycle(proof_ref,prior,old_identity,h._native(native,prior,now=proof['finished_at']),
             now=proof['finished_at'],monotonic_ns=proof['finished_monotonic_ns'])
    template=h._document(config['idle_template_ref']);plan=h._document(template['plan_ref']);command=plan['native_metadata_cli']
    need(command.get('schema')=='native-embedded-scoped-observation-cli/v3'
         and command.get('source_capsule_ref')==value['source_capsule_ref'],'generation_source_cache_not_selected')
    runtime=Path(command['runtime_root'])
    for role,name in (('main_entrypoint','scientific_executor.py'),('health_module','native_executor_health.py'),('owner_source','original_embedded_observation_owner.py'),('driver_source','native_embedded_observation_driver.py')):
        need(profile[role]['path']==str(runtime/'ipfs_accelerate_py/agent_supervisor/rescue'/name),'generation_source_role_root_changed')
    cache=h._document(value['source_capsule_ref'])
    need(value['source_capsule_ref'] in profile['source_descriptors'],'generation_cache_binding_not_source_pinned')
    need(cache.get('schema')=='agent-supervisor/scientific-source-capsule-reuse@1','generation_cache_binding_schema')
    request=h._document(value['cache_initialization_request_ref']);terminal=h._document(value['cache_initialization_terminal_ref'])
    need(terminal.get('schema')=='legal-ir-original-native-control-terminal/v1'
         and terminal.get('request_sha256')==value['cache_initialization_request_ref']['sha256']
         and terminal.get('completed')is True and terminal.get('source_READ_closed')is True
         and terminal.get('original_native_compute_closed')is True and terminal.get('child_reaped')is True
         and type(terminal.get('child_returncode'))is int and terminal['child_returncode']==0
         and terminal.get('storage_status')=='released' and terminal.get('cleanup_errors')==[]
         and terminal.get('received_signals')==[] and terminal.get('partial_or_failed_control_must_be_reconciled')is False,
         'generation_cache_initialization_not_original_closed')
    reservation=terminal.get('reservation',{})
    need(reservation.get('resource_lease',{}).get('released')is True
         and reservation.get('record',{}).get('status')=='released'
         and request.get('source_roots')==[str(ORIGINAL_NAMED_ROOTS[0].parents[1]),'/home/barberb/lift_coding/external/ipfs_accelerate']
         and {key:request.get(key) for key in ('cpu_slots','memory_mb','child_process_slots','storage_bytes','timeout_seconds')}==
             dict(cpu_slots=1,memory_mb=1536,child_process_slots=8,storage_bytes=384*1024**2,timeout_seconds=300),
         'generation_original_native_grants_or_closure_changed')
    cache_root=Path(cache['accepted_pin']['capsule_root'])
    need(Path(request['attempt_directory']) in cache_root.parents
         and root in Path(request['attempt_directory']).parents and root in cache_root.parents
         and any(parent in cache_root.parents for parent in ORIGINAL_NAMED_ROOTS), 'generation_cache_not_original_owned_capture')
    execution=h._document(terminal['execution_request'])
    need(execution.get('request')==value['cache_initialization_request_ref']
         and execution.get('owner_source')==profile['owner_source']
         and execution.get('disk_reservation_id')==reservation.get('reservation_id')==reservation.get('record',{}).get('reservation_id')
         and request.get('source_sha256',{}).get(profile['owner_source']['path'])==profile['owner_source']['sha256']
         and request.get('source_sha256',{}).get(profile['driver_source']['path'])==profile['driver_source']['sha256'],
         'generation_cache_original_source_or_reservation_join')
    custody=terminal.get('sampled_process_custody_final',{})
    need(custody.get('live_processes')==0 and type(custody.get('live_processes'))is int
         and custody.get('live_identities')==[] and custody.get('rss_bytes')==0 and not custody.get('tracking_error')
         and terminal.get('process_spawn_receipts',{}).get('all_recorded_original_births_stopped')is True,
         'generation_cache_original_process_not_closed')
    init_plan_ref=h.descriptor(Path(request['argv'][4]));init_plan=h._document(init_plan_ref)
    need(request['source_sha256'].get(init_plan_ref['path'])==init_plan_ref['sha256'],'generation_cache_original_plan_not_pinned')
    need(init_plan['native_metadata_cli']['schema']=='native-embedded-scoped-observation-cli/v2'
         and init_plan['native_metadata_cli']['import_head_capsule_Git']['source_head']==cache['accepted_pin']['source_head']
         and init_plan['accelerate_root']==command['runtime_root'],
         'generation_cache_not_new_HEAD_initialization')
    outstanding=_released_initialization_record(request,terminal)
    need(outstanding+sum(row['source_bytes'] for row in censuses)
         +check_proof_budget(root,config['max_proof_bytes'],reserve_bytes=REUSE_CYCLE_RESERVE_BYTES,reserve_files=REUSE_CYCLE_RESERVE_FILES)['bytes']
         +REUSE_CYCLE_RESERVE_BYTES<=GLOBAL_RETAINED_BYTES,'generation_global_storage_with_outstanding_claims')
    _recovery_scheduler(request,value['scheduler_snapshot_ref'])
    return prior,proof,spent


def _generation_processes_dead(value,proof,current):
    """Reconcile every recorded original owner; UNKNOWN never grants activation."""
    h=health_owner();prior=h._document(value['prior_profile_snapshot_ref']);root=Path(prior['proof_root'])
    identities=[value['prior_main_identity']]
    for cycle in sorted(root.glob('cycle-*')):
        request_path=cycle/'request.json'
        if request_path.is_file():
            historical=h._document(h.descriptor(request_path))
            _recovery_scheduler(historical,value['scheduler_snapshot_ref'])
        proof_path=cycle/'cycle-proof.json'
        if proof_path.is_file():
            reference=h.descriptor(proof_path);record=h._document(reference)
            identities.extend((record['process'],record['owner_observation']))
            if (cycle/'closed.json').is_file():
                need(h._document(h.descriptor(cycle/'closed.json')).get('cycle_proof_ref')==reference,
                     'generation_historical_closed_proof_changed')
                native=h._document(record['native_diagnostic_ref'])
                h._cycle(reference,prior,value['prior_main_identity'],h._native(native,prior,now=record['finished_at']),
                         now=record['finished_at'],monotonic_ns=record['finished_monotonic_ns'])
            else:
                ack=h._document(h.descriptor(cycle/'closed-idle-reconciliation-v01.json'))
                need(ack.get('schema')=='agent-supervisor/scientific-closed-idle-reconciliation@1'
                     and ack.get('rejected_cycle_proof_ref')==reference,'generation_unknown_historical_cycle')
        elif request_path.is_file():
            ack=h._document(h.descriptor(cycle/'nonstarted-reconciliation.json'))
            need(ack.get('schema')=='agent-supervisor/scientific-nonstarted-reconciliation@1'
                 and ack.get('failed_cycle_root')==str(cycle),'generation_unknown_nonstarted_history')
            authorization=h._document(ack['recovery_ref']);capture=h._document(authorization['activation_capture_ref'])
            old_profile=h._document(authorization['prior_profile_snapshot_ref'])
            old_main=capture['opening_actual_main_identity']
            h._stored_identity(old_main,authorization['prior_profile_ref'],old_profile)
            terminal=original_terminal(h._raw(cycle/'owner.stdout'),h.descriptor(request_path))
            need(terminal.get('control_effects')=='not_started' and terminal.get('storage_status')=='not_requested'
                 and type(terminal.get('owner_birth'))is str and terminal['owner_birth'].isdecimal(),
                 'generation_unknown_original_nonstarted_owner')
            identities.extend((old_main,dict(pid=terminal['owner_pid'],birth_ticks=int(terminal['owner_birth']),boot_id=old_main['boot_id'])))
        elif (cycle/'closed.json').is_file():
            raise ExecutorHold('generation_historical_closed_proof_missing')
    terminal=h._document(value['cache_initialization_terminal_ref'])
    need(type(terminal.get('owner_birth'))is str and terminal['owner_birth'].isdecimal(),
         'generation_cache_initializer_birth_unknown')
    identities.append(dict(pid=terminal['owner_pid'],birth_ticks=int(terminal['owner_birth']),boot_id=terminal['owner_boot_id']))
    for identity in identities:
        need(h._integer(identity.get('pid'),2) and h._integer(identity.get('birth_ticks'),1)
             and h._text(identity.get('boot_id')) and h._definitely_dead_identity(identity,current),
             'generation_original_owner_not_definitely_dead')
    return identities


def authorize_bounded_generation_handover(spec_ref,output):
    """One explicit metadata grant; never start, clear, delete or replay."""
    h=health_owner();spec=h._document(spec_ref)
    h._keys(spec,GENERATION_KEYS-{'recorded_at','manager_before'},'generation_handover_spec')
    profile=h._profile(spec['profile_ref']);config_ref=next(pin for pin in profile['source_descriptors']
        if pin['path']==profile['unit']['argv'][profile['unit']['argv'].index('--config')+1])
    config,profile_ref,_=load_configuration(config_ref)
    manager=h._manager(profile['unit']['name'])
    need(manager.get('MainPID')=='0' and manager.get('ActiveState')in('failed','inactive'),'generation_admin_requires_no_main')
    value=dict(spec,recorded_at=time.time(),manager_before=manager)
    _,proof,_=_validate_generation_handover(value,profile_ref,profile,config)
    _generation_processes_dead(value,proof,dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    path=h._path(str(output));need(Path(profile['proof_root']) in path.parents,'generation_grant_outside_root')
    reference=put(path,value);path.chmod(0o400);return reference


def reconcile_bounded_generation_handover(reference,profile_ref,profile,config,opening_identity):
    h=health_owner();value=h._document(reference)
    need(os.environ.get(GENERATION_SHA_ENV)==reference['sha256'],'generation_grant_not_unit_pinned')
    argv=profile['unit']['argv'];flag='--bounded-generation-handover'
    need(argv.count(flag)==1 and argv[argv.index(flag)+1]==reference['path']
         and not any(flag in argv for flag in ('--known-nonstarted-recovery','--known-closed-idle-recovery')),
         'generation_grant_not_exact_unit_argv')
    need(not h._path(reference['path']).lstat().st_mode&0o222,'generation_grant_not_immutable')
    prior,proof,spent=_validate_generation_handover(value,profile_ref,profile,config)
    _generation_processes_dead(value,proof,opening_identity)
    ack=dict(schema='agent-supervisor/scientific-bounded-generation-ack@1',handover_ref=reference,
        retained_census_ref=value['retained_census_ref'],retained_hold_ref=value['prior_cycle_hold_ref'],
        consuming_main_identity=opening_identity,spent_cycle_count=spent,next_cycle_number=spent+1,
        generation_number=value['generation_number'],max_generations=value['max_generations'])
    path=Path(profile['proof_root'])/'bounded-generation-handover.json'
    if path.exists():need(h._document(h.descriptor(path))==ack,'generation_grant_consumed_by_other_main')
    else:put(path,ack)
    return spent


def serve(config_ref, stop, *, recovery_ref=None, generation_ref=None, catalogue_parser_ref=None):
    h = health_owner(); config, profile_ref, profile = load_configuration(config_ref)
    root = h._path(profile['proof_root']); info = root.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and
        stat.S_IMODE(info.st_mode) == 0o700, 'service_proof_root_not_private')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reconciled_count = 0
        consecutive_admission_failures = 0
        consecutive_catalogue_races = 0
        retained_census_refs = []
        catalogue_parser = None
        need(catalogue_parser_ref is None or (recovery_ref is not None
             and h._document(recovery_ref).get("schema")==CATALOGUE_RACE_SCHEMA),
             "catalogue_parser_requires_explicit_closed_race_mode")
        need(recovery_ref is None or generation_ref is None,'generation_recovery_modes_crossed')
        if generation_ref is not None:
            need(not os.path.lexists(root/'executor-held.json') and not any(root.glob('cycle-*')), 'generation_target_not_fresh')
            opening_identity=require_main_process(profile_ref,profile)
            reconciled_count=reconcile_bounded_generation_handover(generation_ref,profile_ref,profile,config,opening_identity)
            retained_census_refs = [h._document(generation_ref)['retained_census_ref']]
        elif recovery_ref is None:
            need(not os.path.lexists(root / 'executor-held.json'), 'persistent_hold_requires_original_reconciliation')
            need(not any(not (path / 'closed.json').exists() for path in root.glob('cycle-*')), 'unfinished_cycle_requires_original_reconciliation')
            opening_identity = require_main_process(profile_ref, profile)
        else:
            opening_identity = require_main_process(profile_ref, profile)
            kind = h._document(recovery_ref).get('schema')
            if kind == CATALOGUE_RACE_SCHEMA:
                need(catalogue_parser_ref is not None
                     and catalogue_parser_ref==catalogue_parser_reference(profile),
                     "catalogue_parser_explicit_constructor_join")
                _validate_closed_catalogue_race_recovery(h._document(recovery_ref),profile_ref,profile,config)
                catalogue_parser=load_explicit_catalogue_parser(catalogue_parser_ref)
                reconciled_count,retained_census_refs=reconcile_closed_catalogue_race_recovery(
                    recovery_ref,profile_ref,profile,config,opening_identity)
                consecutive_catalogue_races=1
            elif kind == ADMISSION_SCHEMA:
                reconciled_count,consecutive_admission_failures = reconcile_source_admission_recovery(recovery_ref,profile_ref,profile,config,opening_identity)
                retained_generation = h._document(h._document(recovery_ref)['prior_generation_authorization_ref'])
                retained_census_refs = list(retained_generation['retained_generations'])
            elif kind == RECOVERY_SCHEMA:
                reconciled_count = reconcile_known_nonstarted(recovery_ref, profile_ref, profile, config, opening_identity)
            else:
                need(kind == CLOSED_IDLE_SCHEMA, 'unknown_typed_recovery')
                reconciled_count = reconcile_known_closed_idle(recovery_ref, profile_ref, profile, config, opening_identity)
        sequence = 1
        if os.path.lexists(profile['receipt_path']):
            previous = h._document(h.descriptor(profile['receipt_path']))
            if previous.get('identity') == opening_identity:
                sequence = previous['sequence'] + 1
            # A new manager invocation starts sequence1; the canonical helper
            # requires the previous process definitely dead and archives its proof.
        cycle_number = reconciled_count + 1
        for _ in range(config['max_cycles'] - reconciled_count):
            if stop.is_set():
                return 0
            try:
                need(h._document(config_ref) == config and h._profile(profile_ref) == profile,
                    'service_binding_changed')
                need(require_main_process(profile_ref, profile) == opening_identity, 'service_main_identity_changed')
                for retained_census_ref in retained_census_refs:
                    _generation_census(retained_census_ref)
                check_proof_budget(root, config['max_proof_bytes'])
                native = (observe_native(config) if catalogue_parser is None else
                          observe_native(config,catalogue_parser=catalogue_parser))
                template_ref, kind = select_template(config, profile, native)
                if recovery_ref is None and generation_ref is None and cycle_number == sequence:
                    run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop)
                else:
                    options={'cycle_number':cycle_number}
                    if catalogue_parser is not None:options['catalogue_parser']=catalogue_parser
                    run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop, **options)
                consecutive_admission_failures = 0
                consecutive_catalogue_races = 0
                sequence += 1
                cycle_number += 1
                # An approved execution is one finite action. A fresh explicit
                # owner activation is needed before another task dispatch.
                if kind != 'genuine_scoped_idle':
                    persistent_hold(root, root, 'approved_execution_closed_no_automatic_followup')
                    return HOLD_EXIT
            except CatalogueRaceDeferred as deferred:
                consecutive_catalogue_races += 1
                failed_cycle_number=cycle_number
                cycle_number += 1
                if consecutive_catalogue_races>=MAX_CONSECUTIVE_CATALOGUE_RACE_DEFERRALS:
                    persistent_hold(root,Path(str(deferred)).parent,'catalogue_race_consecutive_defer_bound')
                    _recovery_failure(root,failed_cycle_number,sequence,'catalogue_race_consecutive_defer_bound')
                    return HOLD_EXIT
                # A closed negative spends its physical position. The next
                # original observation uses a fresh request; health stays put.
            except SourceAdmissionDeferred as deferred:
                consecutive_admission_failures += 1
                failed_cycle_number = cycle_number
                cycle_number += 1
                if consecutive_admission_failures >= MAX_CONSECUTIVE_SOURCE_ADMISSION_DEFERRALS:
                    persistent_hold(root, Path(str(deferred)).parent, 'source_admission_consecutive_defer_bound')
                    _recovery_failure(root, failed_cycle_number, sequence, 'source_admission_consecutive_defer_bound')
                    return HOLD_EXIT
                # Negative physical attempt consumed; no health sequence/publication.
            except Exception as exc:
                persistent_hold(root, root, str(exc))
                if (recovery_ref is not None or generation_ref is not None) and str(exc)!='proof_projected_storage_bound':
                    _recovery_failure(root, cycle_number, sequence, exc)
                return HOLD_EXIT
            if stop.wait(config['poll_seconds']):
                return 0
        persistent_hold(root, root, 'finite_cycle_budget_exhausted')
        return HOLD_EXIT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['serve'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    recovery_flags = parser.add_mutually_exclusive_group()
    recovery_flags.add_argument('--known-nonstarted-recovery')
    recovery_flags.add_argument('--known-closed-idle-recovery')
    recovery_flags.add_argument('--bounded-generation-handover')
    recovery_flags.add_argument('--known-source-admission-recovery')
    recovery_flags.add_argument('--known-closed-catalogue-race-recovery')
    parser.add_argument('--catalogue-parser-source')
    parser.add_argument('--catalogue-parser-sha256')
    args = parser.parse_args(argv)
    # This is set before any project import, including file-path invocation.
    os.environ['IPFS_ACCEL_SKIP_CORE'] = '1'
    if __package__ in (None, ''):
        producer_root = str(Path(__file__).resolve().parents[3])
        if producer_root in os.environ.get('PYTHONPATH','').split(os.pathsep):
            sys.path.insert(0, producer_root)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        need(sys.dont_write_bytecode and __debug__, 'service_requires_B_and_assertions')
        config_ref = health_owner().descriptor(args.config)
        need(config_ref['sha256'] == args.config_sha256, 'service_config_digest')
        recovery_ref = None
        recovery_path = (args.known_nonstarted_recovery or args.known_closed_idle_recovery
                         or args.known_source_admission_recovery or args.known_closed_catalogue_race_recovery)
        if recovery_path:
            recovery_ref = health_owner().descriptor(recovery_path)
        generation_ref = health_owner().descriptor(args.bounded_generation_handover) if args.bounded_generation_handover else None
        need((args.catalogue_parser_source is None)==(args.catalogue_parser_sha256 is None),
             'catalogue_parser_cli_pair_required')
        if args.catalogue_parser_source is None:
            return serve(config_ref, stop, recovery_ref=recovery_ref, generation_ref=generation_ref)
        parser_ref=health_owner().descriptor(args.catalogue_parser_source)
        need(parser_ref['sha256']==args.catalogue_parser_sha256,'catalogue_parser_cli_digest')
        return serve(config_ref,stop,recovery_ref=recovery_ref,generation_ref=generation_ref,
                     catalogue_parser_ref=parser_ref)
    except Exception as exc:
        print(json.dumps({'schema': 'agent-supervisor/scientific-executor-held@1',
            'reason': str(exc), 'automatic_retry_admitted': False}), file=sys.stderr)
        return HOLD_EXIT


if __name__ == '__main__':
    raise SystemExit(main())
