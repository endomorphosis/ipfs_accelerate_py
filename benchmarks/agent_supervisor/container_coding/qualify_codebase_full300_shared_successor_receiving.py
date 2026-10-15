"""Fresh full300 selected shared successor receiving, with equal-work CPU scans.

This additive producer creates its own sources, native owners and two explicit
setup epochs. Five pages in each mode include three pages in a genuinely fresh
process. All 294 eligible members must produce actual numerical rows; the six
other dispositions remain explicit. Chunk bytes and process observations stay
advisory. Two warm and two cold operation-owned receiving gates are required. Signed
admission, task installation, larger heads, CUDA and proof remain separate gates.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from . import qualify_codebase_full300_companion as custody
from . import qualify_codebase_inventory_resume as native
from . import qualify_codebase_inventory_scan as base
from . import qualify_codebase_signed_successor as deployment
from . import source_successor_full_scan_fixture as shared

MODULE = 'benchmarks.agent_supervisor.container_coding.qualify_codebase_full300_shared_successor_receiving'
SCHEMA = 'full300-paired-shared-successor-native-qualification@1'
RESUME_REQUEST = 'full300-paired-shared-successor-resume-request@1'
RESUME_RESULT = 'full300-paired-shared-successor-resume-result@1'
COLD_REQUEST = 'full300-paired-shared-successor-cold-request@1'
COLD_RESULT = 'full300-paired-shared-successor-cold-result@1'
COLD_LIFECYCLE = 'full300-paired-shared-successor-cold-owned-lifecycle@1'
COLD_RSS = 'full300-paired-shared-successor-cold-rss@1'
COLD_REQUEST_FIELDS = frozenset({'schema', 'output', 'mode', 'selection_cid', 'source_delta_cid',
    'root_cid', 'completion_cid', 'pages', 'versions', 'checkpoint_states', 'expected_head',
    'authority', 'authority_sha256', 'orchestration', 'parent_identity', 'result', 'rss_result',
    'timeout_seconds', 'generation_inputs_sha256'})
COLD_RESULT_FIELDS = frozenset({'schema', 'qualified', 'full_campaign_qualified', 'pid',
    'process_identity', 'parent_identity', 'mode', 'request_sha256', 'authority_sha256',
    'orchestration_lease_id', 'generation_inputs_sha256', 'selection_cid', 'source_delta_cid',
    'root_cid', 'completion_cid', 'pages', 'versions', 'expected_head', 'checkpoint_states_before',
    'checkpoint_states_after', 'variant_heads_before', 'variant_heads_after',
    'registry_owner_before', 'registry_owner_after', 'paired_receiver_phase_record',
    'post_setup_fit_attempt_count', 'new_fitting_epochs', 'receive_forward_attempt_count',
    'receiving_extra_numerical_attempt_count', 'receiving_mutation_attempt_count',
    'no_install_guards', 'rss_observation_file', 'rss_observation_pin',
    'selected_producers_unchanged', 'native_owners_closed', 'resources_after', 'cleanup_errors',
    'proof_authority', 'cuda_qualified', '384d_qualified', '768d_qualified', '4096d_qualified'})
REPOSITORY_ID = 'qualification:source-successor'
ENTRIES, PAGE_ENTRIES, PAGES, PARENT_PAGES, RESUME_PAGES = 300, 64, 5, 2, 3
EXPECTED_DISPOSITIONS = {'inferred': 294, 'opaque': 2, 'parse_failed': 1,
                         'unindexed': 1, 'unsupported_target': 2}
EXPECTED_PAGE_ROWS = [63, 64, 64, 64, 39]
MODES = ('legacy_default', 'default', 'reference')


def _protocol(mode):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as old_scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunks as old_records
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_scan as new_scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_chunks as new_records
    native.require(mode in MODES, 'explicit fresh qualification mode required')
    if mode == 'legacy_default':
        return old_scan, old_records
    return SimpleNamespace(
        start_current_codebase_chunked_scan=new_scan.start_current_codebase_shared_scan,
        scan_current_codebase_page_chunked=new_scan.scan_current_codebase_page_shared,
        complete_current_codebase_chunked_scan=new_scan.complete_current_codebase_shared_scan,
        validate_current_codebase_chunked_scan_page=new_scan.validate_current_codebase_shared_scan_page,
        validate_current_codebase_chunked_scan_completion=new_scan.validate_current_codebase_shared_scan_completion,
        load_codebase_scan_numerical_chunk=new_scan.load_codebase_scan_shared_numerical_chunk,
        load_codebase_scan_chunked_page=new_scan.load_codebase_scan_shared_page,
        load_codebase_scan_chunked_completion=new_scan.load_codebase_scan_shared_completion), new_records


class _ParentRSSMonitor:
    """Bounded producer RSS samples; no process discovery or memory estimate."""

    def __init__(self):
        from ipfs_datasets_py.logic.backends import process
        self.process, self.started = process, time.monotonic()
        self.identity = process._linux_owned_identity(os.getpid())[0].to_dict()
        self.record = {'identity': self.identity, 'sample_count': 0, 'peak_rss_bytes': 0,
            'first_elapsed_seconds': None, 'last_elapsed_seconds': None,
            'sample_interval_seconds': .05, 'monitor_joined': False, 'errors': []}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='full300-shared-parent-rss', daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                identity = self.process._linux_owned_identity(os.getpid())[0].to_dict()
                observed = self.process._linux_process_identity(os.getpid())
                native.require(identity == self.identity and observed is not None
                    and observed[0] == identity['birth_ticks'], 'parent RSS birth identity changed')
                rss = observed[3] * os.sysconf('SC_PAGE_SIZE')
                native.require(type(rss) is int and rss > 0, 'positive actual parent RSS sample required')
                elapsed = time.monotonic() - self.started
                self.record['sample_count'] += 1
                self.record['peak_rss_bytes'] = max(self.record['peak_rss_bytes'], rss)
                if self.record['first_elapsed_seconds'] is None:
                    self.record['first_elapsed_seconds'] = elapsed
                self.record['last_elapsed_seconds'] = elapsed
            except BaseException as error:
                if len(self.record['errors']) < 32:
                    self.record['errors'].append({'error_type': type(error).__name__, 'error': str(error)[:2048]})
            if self.stop.wait(.05):
                return

    def finish(self):
        self.stop.set()
        self.thread.join(5)
        self.record['monitor_joined'] = not self.thread.is_alive()
        native.require(self.record['monitor_joined'] and self.record['sample_count'] > 0
            and not self.record['errors'], 'actual parent RSS monitor did not finish cleanly')
        return self.record


class _RSSObserver:
    """Observe real native guards and preserve their exact returned values.

    Owned native runners sample their birth-tracked resident-byte method, so
    both that method and the ordinary process-tree helper are observed. The
    latter is a supplemental actual sample; it never replaces a guard value.
    Only successful numerical runner leaders enter the exported population.
    """

    def __init__(self, output, mode):
        from ipfs_datasets_py.logic.backends import process
        self.output, self.mode, self.process = output, mode, process
        self.started, self.samples, self.children = time.monotonic(), {}, []
        self.errors = []
        self.parent = None
        self.stack = ExitStack()

    def _sample(self, leader, value, tree):
        if type(value) is not int or value <= 0:
            return
        key = (leader['pid'], leader['birth_ticks'])
        native.require(key in self.samples or len(self.samples) < 64,
                       'bounded observed native leader population exceeded')
        record = self.samples.setdefault(key, {'leader': leader, 'sample_count': 0,
            'peak_rss_bytes': 0, 'first_elapsed_seconds': None, 'last_elapsed_seconds': None,
            'tree_sample_count': 0, 'tree_peak_rss_bytes': 0})
        native.require(record['leader'] == leader, 'native RSS leader identity changed')
        if tree:
            record['tree_sample_count'] += 1
            record['tree_peak_rss_bytes'] = max(record['tree_peak_rss_bytes'], value)
        else:
            elapsed = time.monotonic() - self.started
            record['sample_count'] += 1
            record['peak_rss_bytes'] = max(record['peak_rss_bytes'], value)
            if record['first_elapsed_seconds'] is None:
                record['first_elapsed_seconds'] = elapsed
            record['last_elapsed_seconds'] = elapsed

    def _observation_error(self, stage, error):
        if len(self.errors) < 32:
            self.errors.append({'stage': stage, 'error_type': type(error).__name__, 'error': str(error)[:2048]})

    def __enter__(self):
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as old_scan
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_scan as new_scan
        real_tree = self.process._process_tree_resident_bytes
        real_tracked = self.process._LinuxDescendantTracker.resident_bytes

        def tree(leader_pid):
            before = None
            try:
                before = self.process._linux_owned_identity(leader_pid)[0].to_dict()
            except (FileNotFoundError, ProcessLookupError):
                pass
            except BaseException as error:
                self._observation_error('tree_identity_before', error)
            value = real_tree(leader_pid)
            if before is None:
                return value
            try:
                after = self.process._linux_owned_identity(leader_pid)[0].to_dict()
                if before == after:
                    self._sample(before, value, True)
            except (FileNotFoundError, ProcessLookupError):
                pass
            except BaseException as error:
                self._observation_error('tree_identity_after_or_sample', error)
            return value

        def tracked(tracker):
            # The exact native guard result is computed and returned unchanged.
            value = real_tracked(tracker)
            try:
                before = self.process._linux_owned_identity(tracker.leader_pid)[0].to_dict()
                self.process._process_tree_resident_bytes(tracker.leader_pid)
                after = self.process._linux_owned_identity(tracker.leader_pid)[0].to_dict()
                if before == after:
                    self._sample(before, value, False)
            except (FileNotFoundError, ProcessLookupError):
                pass
            except BaseException as error:
                self._observation_error('tracked_supplement_or_sample', error)
            return value

        def runner_for(real_runner):
            def observe(command, raw, *args, **kwargs):
                numerical = isinstance(command, (tuple, list)) and any(
                    Path(str(value)).name in {'codebase_inventory_resume_worker.py',
                                             'codebase_inventory_shared_worker.py'} for value in command)
                file_name = None
                if numerical:
                    native.require(type(raw) is bytes and len(raw) <= 32 * 1024 * 1024,
                                   'actual bounded numerical stdin required')
                    if self.mode != 'legacy_default':
                        value = json.loads(raw)
                        native.require(value['schema'] == 'codebase-inventory-resume-shared-worker@1'
                            and type(value['member_indices']) is list and value['member_indices'],
                            'actual shared numerical stdin profile differs')
                        file_name = self.mode + '-worker-input-' + str(value['member_indices'][0]).zfill(4) + '.json'
                        path = self.output / file_name
                        with path.open('xb') as stream:
                            stream.write(raw)
                            stream.flush()
                            os.fsync(stream.fileno())
                        path.chmod(0o444)
                        del value
                result = real_runner(command, raw, *args, **kwargs)
                if numerical:
                    lifecycle = result.native_lifecycle
                    native.require(lifecycle is not None and lifecycle.cleanup_confirmed,
                                   'RSS population needs an actually drained numerical run')
                    leader = lifecycle.leader.to_dict()
                    sample = self.samples.get((leader['pid'], leader['birth_ticks']))
                    native.require(sample is not None and sample['leader'] == leader
                        and sample['sample_count'] > 0 and sample['tree_sample_count'] > 0,
                        'numerical leader has no positive actual birth-bound RSS samples')
                    bound = kwargs['limits'].resident_memory_bytes
                    native.require(type(bound) is int and 0 < sample['peak_rss_bytes'] <= bound
                        and len(self.children) < 64, 'actual numerical sampled RSS exceeded native bound')
                    self.children.append({**sample, 'resident_memory_limit_bytes': bound,
                        'actual_runner_returned': True, 'cleanup_confirmed': True, 'input_file': file_name,
                        'input_sha256': hashlib.sha256(raw).hexdigest(), 'input_bytes': len(raw)})
                return result
            return observe

        self.parent = _ParentRSSMonitor()
        try:
            self.stack.enter_context(patch.object(self.process, '_process_tree_resident_bytes', tree))
            self.stack.enter_context(patch.object(self.process._LinuxDescendantTracker, 'resident_bytes', tracked))
            for scan in (old_scan, new_scan):
                self.stack.enter_context(patch.object(scan, 'run_bounded_stdin_tool', runner_for(scan.run_bounded_stdin_tool)))
        except BaseException:
            try:
                self.stack.close()
            finally:
                self.parent.finish()
            raise
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.stack.close()
        finally:
            self.parent_record = self.parent.finish()
        native.require(not self.errors, 'actual RSS observation failed: ' + str(self.errors))

    def document(self):
        native.require(hasattr(self, 'parent_record'), 'RSS monitor must join before report publication')
        return {'schema': 'full300-shared-process-rss-observation@1', 'mode': self.mode,
            'owner_identity': self.parent.identity, 'parent': self.parent_record,
            'numerical_children': self.children, 'max_observed_leaders': 64,
            'observation_errors': self.errors,
            'sampled_guard_values_returned_unchanged': True, 'kernel_memory_attestation': False}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _persist(output, name, record):
    custody._persist(output, name, record)


def _states(registry, versions):
    return custody._states(registry, versions)


def _launch_resume(command, log, timeout, orchestration):
    """Hold the launch lease until the sampled leader and descendants drain.

    Numerical grandchildren start separate sessions. The new owned native
    runner tracks their birth identities through ancestry rather than relying
    on the resume leader's process group. The enclosing dedicated container
    remains the kernel containment boundary at its independent outer deadline.
    """
    from ipfs_datasets_py.logic.backends.process import (
        BoundedToolRunner, OwnedToolCleanupError, ToolRunLimits, run_bounded_stdin_tool,
    )
    runner = BoundedToolRunner(base_environment=dict(os.environ))
    bounds = ToolRunLimits(timeout_seconds=timeout, termination_grace_seconds=1,
        max_input_bytes=1, max_output_bytes=16 * 1024 * 1024,
        max_workspace_bytes=128 * 1024 * 1024, max_file_bytes=64 * 1024 * 1024,
        resident_memory_bytes=8 * 1024 * 1024 * 1024)
    try:
        process = run_bounded_stdin_tool(command, b'', runner=runner, limits=bounds,
            cancellation=orchestration.combined_cancellation_signal(None), owned_process_observation=True)
    except OwnedToolCleanupError as error:
        # No caller context exit while cleanup ownership is unresolved. Each
        # attempt remains bounded; the parent process and host container lease
        # stay alive until actual cleanup or the outer container guardian.
        attempts, begin, last_notice = 0, time.monotonic(), 0.0
        while True:
            attempts += 1
            cleanup = None
            try:
                orchestration.renew()
            except BaseException:
                pass
            try:
                cleanup = error.cleanup_handle.retry_cleanup()
            except BaseException:
                pass
            if cleanup is not None and cleanup.cleanup_confirmed:
                log.write(_wire({'status': 'fresh_resume_cleanup_confirmed_after_failed_run',
                    'attempts': attempts, 'elapsed_seconds': time.monotonic() - begin,
                    'native_lifecycle': cleanup.to_dict()}) + b'\n')
                log.flush()
                os.fsync(log.fileno())
                raise
            now = time.monotonic()
            if now - last_notice >= 5:
                last_notice = now
                try:
                    notice = {'status': 'fresh_resume_launch_lease_retained_until_actual_cleanup',
                        'attempts': attempts, 'elapsed_seconds': now - begin,
                        'native_lifecycle': None if cleanup is None else cleanup.to_dict()}
                    log.write(_wire(notice) + b'\n')
                    log.flush()
                    print(json.dumps({'status': notice['status'], 'attempts': attempts}, sort_keys=True), flush=True)
                except BaseException:
                    pass
            try:
                time.sleep(.25)
            except BaseException:
                pass
    lifecycle = process.native_lifecycle
    native.require(lifecycle is not None and lifecycle.cleanup_confirmed
        and lifecycle.parent_pid == os.getpid() and lifecycle.leader.parent_pid == os.getpid(),
        'fresh resume has no actual parent-owned drained lifecycle')
    leader = lifecycle.leader
    identity = {'pid': leader.pid, 'birth': str(leader.birth_ticks), 'parent_pid': leader.parent_pid,
        'group_pid': leader.process_group_id, 'session_pid': leader.session_id, 'state': 'reaped'}
    native.require(identity['group_pid'] == identity['session_pid'] == identity['pid'],
                   'fresh resume leader did not own its isolated native session')
    cleanup = {'schema': 'full300-shared-resume-owned-lifecycle@1',
        'launch_leader_reaped': lifecycle.leader_reaped,
        'live_group_members': len(lifecycle.remaining_group_members),
        'live_tracked_descendants': len(lifecycle.remaining_tracked_descendants),
        'group_pid': leader.process_group_id, 'session_pid': leader.session_id,
        'native_lifecycle': lifecycle.to_dict(), 'sampled_ownership_only': True,
        'kernel_containment_attested': False,
        'scope': 'native sampled ancestry and birth-bound descendants; launch lease held until actual cleanup',
        'runner': {'returncode': process.returncode, 'timed_out': process.timed_out,
            'cancelled': process.cancelled, 'unavailable': process.unavailable,
            'resource_exhausted': process.resource_exhausted, 'output_truncated': process.output_truncated,
            'workspace_cleaned': process.workspace_cleaned, 'elapsed_ms': process.elapsed_ms,
            'max_output_bytes': bounds.max_output_bytes, 'max_workspace_bytes': bounds.max_workspace_bytes,
            'max_file_bytes': bounds.max_file_bytes, 'resident_memory_bytes': bounds.resident_memory_bytes}}
    log.write(process.stdout.encode('utf-8'))
    log.write(process.stderr.encode('utf-8'))
    log.write(b'\n' + _wire({'parent_owned_launch_lifecycle': cleanup}) + b'\n')
    log.flush()
    os.fsync(log.fileno())
    native.require(not any((process.timed_out, process.cancelled, process.unavailable,
        process.resource_exhausted, process.output_truncated)), 'fresh resume native runner exceeded its bounds')
    return process.returncode, identity, cleanup


def _export_page(output, mode, number, page, index):
    """Persist every actual chunk separately; retain only bounded descriptors."""
    scan, records = _protocol(mode)
    _persist(output, mode + '-page-' + str(number) + '.json', page)
    p, previous, rows, projections = page.to_dict(), None, 0, 0
    for ordinal, descriptor in enumerate(p['chunks']):
        chunk = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid'])
        c = chunk.to_dict()
        native.require(records.chunk_descriptor(chunk) == descriptor and c['chunk_ordinal'] == ordinal
            and c['previous_chunk_cid'] == previous and c['worker_counters']['training_calls'] == 0
            and c['worker_counters']['rows'] == len(c['inference']['rows']) == descriptor['rows'],
            'actual full300 chunk mapping, prefix or worker counters differ')
        native.require(c['worker_receipt']['input_bytes'] <= 32 * 1024 * 1024
            and c['worker_receipt']['output_bytes'] <= 16 * 1024 * 1024
            and len(chunk._payload) <= 16 * 1024 * 1024,
            'full300 actual request, response or wrapper exceeds unchanged cap')
        for row_index, member_index in enumerate(c['member_indices']):
            entry = p['entries'][member_index - p['start']]
            native.require(entry['chunk_ordinal'] == ordinal and entry['row_index'] == row_index
                and entry['target_sha256'] == c['target_sha256s'][row_index]
                and entry['source_digest'] == c['source_digests'][row_index]
                == c['inference']['rows'][row_index]['source_digest'], 'actual full300 row mapping differs')
        name = mode + '-page-' + str(number) + '-chunk-' + str(ordinal + 1) + '.json'
        _persist(output, name, chunk)
        previous = chunk.artifact_cid
        rows += descriptor['rows']
        projections += descriptor['projection_records']
        del chunk, c
    native.require(rows == p['coverage']['inferred_rows'] == EXPECTED_PAGE_ROWS[number - 1]
        and projections == p['coverage']['projection_records'] == 2 * rows
        and p['coverage']['chunks'] == len(p['chunks']), 'actual full300 page coverage differs')
    return records.page_descriptor(page)


def _verify_page_exports(output, mode, number, page, index):
    """Join fresh-child ordinary exports back to the independently read CAS."""
    scan, _ = _protocol(mode)
    name = mode + '-page-' + str(number) + '.json'
    saved = json.loads(shared.read_absolute(output / name, 32 * 1024 * 1024))
    native.require(shared.inert.observation_same(saved,
        {'artifact_cid': page.artifact_cid, 'value': page.to_dict()}), 'fresh child page export bytes differ')
    for ordinal, descriptor in enumerate(page.to_dict()['chunks']):
        chunk = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid'])
        name = mode + '-page-' + str(number) + '-chunk-' + str(ordinal + 1) + '.json'
        saved = json.loads(shared.read_absolute(output / name, 32 * 1024 * 1024))
        native.require(shared.inert.observation_same(saved,
            {'artifact_cid': chunk.artifact_cid, 'value': chunk.to_dict()}), 'fresh child chunk export bytes differ')
        del chunk, saved


@contextmanager
def _no_forward(report):
    """Receiving replays native targets; no numerical child, forward or fit."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_scan as shared_scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_feature_worker as numerical
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_resume_worker as old_worker
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_shared_worker as new_worker
    real_popen = subprocess.Popen

    def refuse(*args, **kwargs):
        report['receive_forward_attempt_count'] += 1
        raise AssertionError('chunk receiving attempted numerical inference')

    def popen(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and any(
                any(name in str(value) for name in ('codebase_inventory_resume_worker.py',
                    'codebase_inventory_shared_worker.py')) for value in command):
            return refuse(command, *args, **kwargs)
        return real_popen(command, *args, **kwargs)

    with ExitStack() as stack:
        for owner, name in ((scan, '_worker'), (shared_scan, '_worker'), (resume, '_worker'),
                (old_worker, 'execute'), (new_worker, 'execute'),
                (numerical._PreparedProjectionEngine, 'infer'),
                (features, 'infer_projection_features'), (runtimes.SourceBoundCodebaseFeatureRuntime, 'infer')):
            stack.enter_context(patch.object(owner, name, refuse))
        stack.enter_context(patch.object(subprocess, 'Popen', popen))
        yield


RECEIVER_PROFILE = 'paired-shared512-successor-native-receiving@1'
RECEIVE_PHASE_SCHEMA = 'full300-paired-shared-successor-receive-phase@1'
EXPECTED_RECEIVE_DISPOSITIONS = {
    'inferred': 294, 'opaque': 2, 'parse_failed': 1,
    'unindexed': 1, 'unsupported_target': 2,
}


def _require_receive(condition, message):
    if not condition:
        raise AssertionError(message)


def _start_selected_shared_root(mode, index, repository, *, source_delta,
        registry, previous_version_id, version_id, options):
    """Create a genuine selected direct-child successor, then load its root."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
    _require_receive(mode in ('default', 'reference'), 'shared successor mode required')
    selection = successor.start_current_codebase_successor_scan(
        index, repository, source_delta=source_delta, registry=registry,
        previous_version_id=previous_version_id, version_id=version_id,
        limits=resume.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=64),
        optimized=mode == 'default', **options)
    root = resume.load_codebase_scan_resume_root(index.artifacts, selection.to_dict()['root_cid'])
    value = selection.to_dict()
    _require_receive(value['source_delta_cid'] == source_delta.artifact_cid
        and value['root_cid'] == root.artifact_cid
        and value['model']['version_id'] == version_id
        and value['previous_model']['version_id'] == previous_version_id
        and value['optimized'] is (mode == 'default')
        and root.to_dict()['optimized'] is value['optimized']
        and len(root.to_dict()['members']) == 300,
        'actual selected successor/root/parent/child population differs')
    return selection, root


@contextmanager
def _receiving_guards(report, *, no_forward_guard, no_fit_guard):
    """Guard real entry points only during receiving, after setup training.

    Existing benchmark guards remain active. Extra guards cover direct native
    runtime/feature-worker paths and the actual source-training worker. Each
    intercepted call increments a counter and refuses before dispatch. These
    guards give boundary instrumentation; they do not attest arbitrary code.
    """
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_scan as shared_scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as old_scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_feature_worker as feature_worker
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_feature_worker as numerical
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_shared_worker as shared_worker
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_resume_worker as resume_worker
    report.setdefault('receiving_extra_numerical_attempt_count', 0)
    report.setdefault('receiving_mutation_attempt_count', 0)

    def refuse(*args, **kwargs):
        report['receiving_extra_numerical_attempt_count'] += 1
        raise AssertionError('paired receiving attempted numerical forward/fitting/dispatch')

    def refuse_mutation(*args, **kwargs):
        report['receiving_mutation_attempt_count'] += 1
        raise AssertionError('paired receiving attempted source/artifact publication or another scan')

    original_popen = subprocess.Popen
    worker_stems = ('codebase_feature_worker', 'codebase_inventory_feature_worker',
        'codebase_inventory_resume_worker', 'codebase_inventory_shared_worker',
        'codebase_feature_training_worker')
    worker_files = {stem + '.py' for stem in worker_stems}
    worker_modules = {'ipfs_datasets_py.optimizers.logic_theorem_optimizer.' + stem
        for stem in worker_stems}

    def guarded_popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and any(str(value) in worker_modules
                or os.path.basename(str(value)) in worker_files for value in command):
            return refuse(command, *args, **kwargs)
        return original_popen(command, *args, **kwargs)

    with no_forward_guard(report), no_fit_guard(report), ExitStack() as stack:
        for owner, name in ((training, '_worker'), (training, 'train_current_codebase_features'),
                (shared_scan, '_worker'), (resume, '_worker'),
                (shared_worker, 'execute'), (resume_worker, 'execute'),
                (feature_worker, 'execute'), (numerical, 'execute'), (runtimes.NativeRuntime, 'train'),
                (runtimes.NativeRuntime, 'infer')):
            stack.enter_context(patch.object(owner, name, refuse))
        for owner, name in ((ImmutableCAS, 'put'), (ImmutableCAS, 'put_bytes'),
                (RepositoryCodebaseIndex, 'prepare_current'),
                (successor, 'start_current_codebase_successor_scan'),
                (shared_scan, 'start_current_codebase_shared_scan'),
                (shared_scan, 'scan_current_codebase_page_shared'),
                (shared_scan, 'complete_current_codebase_shared_scan'),
                (old_scan, 'start_current_codebase_chunked_scan'),
                (old_scan, 'scan_current_codebase_page_chunked'),
                (old_scan, 'complete_current_codebase_chunked_scan')):
            stack.enter_context(patch.object(owner, name, refuse_mutation))
        stack.enter_context(patch.object(subprocess, 'Popen', guarded_popen))
        yield


def _variant_heads(registry, versions):
    result = {}
    for label, version_id in versions.items():
        value = registry.get_version(version_id)
        result[label] = {'version_id': version_id, 'variant_id': value['variant_id'],
            'main_head': registry.resolve_head(value['variant_id'], 'main')}
    return result


def _registry_owner(registry):
    value = {'pid': registry._pid, 'owner_generation': registry.owner_generation}
    native.require(type(value['pid']) is int and value['pid'] == os.getpid()
        and type(value['owner_generation']) is int and value['owner_generation'] > 0,
        'actual native registry owner observation required')
    return value


def _receive_selected_pair(mode, stage, selection, completion, index, repository,
        *, root, registry, versions, frozen_states, state_reader, options,
        late_callback, no_forward_guard, no_fit_guard, rss_monitor_factory,
        process_identity_reader, report):
    """Accept one exact saved pair only after callback, close and scope exit.

    rss_monitor_factory is the real producer's _ParentRSSMonitor, which samples
    this process identity and joins its sampling thread at finish. Its bound
    here is the 8 GiB container ceiling. The 1024 MiB operation reservation is
    serialized retention accounting; it is not a 1024 MiB process RSS claim.
    The cold child must call the same helper after fresh _open and CAS loads.
    """
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_successor_receiving as receiving
    _require_receive(mode in ('default', 'reference') and stage in ('warm', 'cold'),
        'fixed paired receiver mode/stage required')
    _require_receive(type(options['memory_mb']) is int and options['memory_mb'] == 1024,
        'explicit unchanged serialized operation reservation required')
    _require_receive(selection.to_dict()['root_cid'] == root.artifact_cid
        and completion.to_dict()['root_cid'] == root.artifact_cid
        and selection.to_dict()['optimized'] is (mode == 'default')
        and root.to_dict()['optimized'] is (mode == 'default'),
        'selected root/completion/profile join differs')
    coverage = completion.to_dict()['coverage']
    _require_receive(coverage['inventory_entries'] == 300 and coverage['pages'] == 5
        and coverage['inferred_rows'] == 294 and coverage['projection_records'] == 588
        and coverage['dispositions'] == EXPECTED_RECEIVE_DISPOSITIONS,
        'saved full300 coverage differs')
    before = state_reader(registry, versions)
    _require_receive(before == frozen_states, 'weights/Adam/report differ before receiving')
    heads_before, owner_before = _variant_heads(registry, versions), _registry_owner(registry)
    counters = ('post_setup_fit_attempt_count', 'receive_forward_attempt_count',
        'receiving_extra_numerical_attempt_count', 'receiving_mutation_attempt_count')
    for counter in counters:
        report.setdefault(counter, 0)
    attempt_counts_before = {name: report[name] for name in counters}
    _require_receive(all(type(value) is int and value == 0
        for value in attempt_counts_before.values()), 'preexisting numerical receiving attempt')
    started = time.monotonic()
    identity = process_identity_reader(os.getpid())
    _require_receive(identity is not None and identity['pid'] == os.getpid(),
        'actual invoking process identity required')
    row = {'schema': RECEIVE_PHASE_SCHEMA, 'profile': RECEIVER_PROFILE,
        'mode': mode, 'stage': stage, 'accepted': False, 'full_campaign_qualified': False,
        'invoking_process_identity': identity,
        'selection_cid': selection.artifact_cid, 'root_cid': root.artifact_cid,
        'completion_cid': completion.artifact_cid,
        'source_delta_cid': selection.to_dict()['source_delta_cid'], 'coverage': coverage,
        'callback_invocations': 0, 'successful_close_calls': 0,
        'native_scope_exit_success': False, 'operation_reservation_mb': 1024,
        'sampled_rss_bound_bytes': 8 * 1024**3,
        'checkpoint_states_before': before,
        'variant_heads_before': heads_before, 'registry_owner_before': owner_before,
        'authority': {'proof': False, 'signed_admission': False, 'supervisor_dispatch': False,
            'cuda': False, '384d': False, '768d': False, '4096d': False},
        'sampled_memory_observation_only': True, 'kernel_memory_attestation': False}
    monitor = rss_monitor_factory()
    try:
        with _receiving_guards(report,
                no_forward_guard=no_forward_guard, no_fit_guard=no_fit_guard):
            with receiving._paired_current_codebase_shared_successor_completion(
                    selection, completion, index, repository, root=root, registry=registry,
                    **options) as close_current:
                row['entry_returned_elapsed_seconds'] = time.monotonic() - started
                # Callbacks receive immutable records, never the private gate.
                late_callback(selection, completion)
                row['callback_invocations'] += 1
                row['callback_completed_elapsed_seconds'] = time.monotonic() - started
                actual = close_current()
                _require_receive(actual is completion, 'paired close returned another object')
                row['successful_close_calls'] += 1
                row['close_returned_elapsed_seconds'] = time.monotonic() - started
            row['native_scope_exit_success'] = True
            row['scope_exit_elapsed_seconds'] = time.monotonic() - started
        after = state_reader(registry, versions)
        _require_receive(after == frozen_states, 'weights/Adam/report changed during receiving')
        row['checkpoint_states_after'] = after
        row['variant_heads_after'], row['registry_owner_after'] = _variant_heads(registry, versions), _registry_owner(registry)
        _require_receive(row['variant_heads_after'] == heads_before and row['registry_owner_after'] == owner_before,
            'receiving changed variant heads or its current native owner generation')
        row['numerical_attempt_counts'] = {name: report[name] for name in counters}
        _require_receive(row['numerical_attempt_counts'] == attempt_counts_before,
            'receiving performed a forward/fit/dispatch attempt')
        _require_receive(row['callback_invocations'] == row['successful_close_calls'] == 1,
            'paired receiving callback/one-use close accounting differs')
        row['accepted'] = True
        return row
    except BaseException as error:
        row.update(accepted=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # A real observation failure makes this phase fail; successful native
        # close alone is insufficient to qualify the future complete campaign.
        primary = sys.exc_info()[0] is not None
        memory_error = None
        try:
            row['rss_observation'] = monitor.finish()
            memory = row['rss_observation']
            observed_identity = memory['identity']
            # Keep the producer's custody identity and the real monitor's
            # NativeProcessIdentity serialization in their original formats.
            identity_matches = (observed_identity['pid'] == identity['pid']
                and observed_identity['birth_ticks'] == int(identity['birth'])
                and observed_identity['parent_pid'] == identity['parent_pid']
                and observed_identity['process_group_id'] == identity['group_pid']
                and observed_identity['session_id'] == identity['session_pid'])
            _require_receive(memory['monitor_joined'] is True and not memory['errors']
                and identity_matches
                and type(memory['sample_count']) is int and memory['sample_count'] > 0
                and type(memory['peak_rss_bytes']) is int
                and 0 < memory['peak_rss_bytes'] <= row['sampled_rss_bound_bytes'],
                'actual joined positive receiving RSS observation required')
        except BaseException as error:
            row.update(accepted=False, rss_error_type=type(error).__name__, rss_error=str(error))
            memory_error = error
        row['elapsed_seconds'] = time.monotonic() - started
        report.setdefault('paired_receiver_phase_records', []).append(row)
        if memory_error is not None and not primary:
            raise memory_error




def _launch_cold_receiver(command, log, timeout, orchestration):
    """Hold the cold launch lease until its actual leader and descendants drain.

    Numerical grandchildren start separate sessions. The new owned native
    runner tracks their birth identities through ancestry rather than relying
    on the resume leader's process group. The enclosing dedicated container
    remains the kernel containment boundary at its independent outer deadline.
    """
    from ipfs_datasets_py.logic.backends.process import (
        BoundedToolRunner, OwnedToolCleanupError, ToolRunLimits, run_bounded_stdin_tool,
    )
    runner = BoundedToolRunner(base_environment=dict(os.environ))
    bounds = ToolRunLimits(timeout_seconds=timeout, termination_grace_seconds=1,
        max_input_bytes=1, max_output_bytes=16 * 1024 * 1024,
        max_workspace_bytes=128 * 1024 * 1024, max_file_bytes=64 * 1024 * 1024,
        resident_memory_bytes=8 * 1024 * 1024 * 1024)
    try:
        process = run_bounded_stdin_tool(command, b'', runner=runner, limits=bounds,
            cancellation=orchestration.combined_cancellation_signal(None), owned_process_observation=True)
    except OwnedToolCleanupError as error:
        # No caller context exit while cleanup ownership is unresolved. Each
        # attempt remains bounded; the parent process and host container lease
        # stay alive until actual cleanup or the outer container guardian.
        attempts, begin, last_notice = 0, time.monotonic(), 0.0
        while True:
            attempts += 1
            cleanup = None
            try:
                orchestration.renew()
            except BaseException:
                pass
            try:
                cleanup = error.cleanup_handle.retry_cleanup()
            except BaseException:
                pass
            if cleanup is not None and cleanup.cleanup_confirmed:
                log.write(_wire({'status': 'fresh_cold_receiver_cleanup_confirmed_after_failed_run',
                    'attempts': attempts, 'elapsed_seconds': time.monotonic() - begin,
                    'native_lifecycle': cleanup.to_dict()}) + b'\n')
                log.flush()
                os.fsync(log.fileno())
                raise
            now = time.monotonic()
            if now - last_notice >= 5:
                last_notice = now
                try:
                    notice = {'status': 'fresh_cold_receiver_launch_lease_retained_until_actual_cleanup',
                        'attempts': attempts, 'elapsed_seconds': now - begin,
                        'native_lifecycle': None if cleanup is None else cleanup.to_dict()}
                    log.write(_wire(notice) + b'\n')
                    log.flush()
                    print(json.dumps({'status': notice['status'], 'attempts': attempts}, sort_keys=True), flush=True)
                except BaseException:
                    pass
            try:
                time.sleep(.25)
            except BaseException:
                pass
    lifecycle = process.native_lifecycle
    native.require(lifecycle is not None and lifecycle.cleanup_confirmed
        and lifecycle.parent_pid == os.getpid() and lifecycle.leader.parent_pid == os.getpid()
        and process.workspace_cleaned,
        'fresh cold receiver has no actual parent-owned drained lifecycle')
    leader = lifecycle.leader
    identity = {'pid': leader.pid, 'birth': str(leader.birth_ticks), 'parent_pid': leader.parent_pid,
        'group_pid': leader.process_group_id, 'session_pid': leader.session_id, 'state': 'reaped'}
    native.require(identity['group_pid'] == identity['session_pid'] == identity['pid'],
                   'fresh cold receiver leader did not own its isolated native session')
    cleanup = {'schema': COLD_LIFECYCLE,
        'launch_leader_reaped': lifecycle.leader_reaped,
        'live_group_members': len(lifecycle.remaining_group_members),
        'live_tracked_descendants': len(lifecycle.remaining_tracked_descendants),
        'group_pid': leader.process_group_id, 'session_pid': leader.session_id,
        'native_lifecycle': lifecycle.to_dict(), 'sampled_ownership_only': True,
        'kernel_containment_attested': False,
        'scope': 'native sampled ancestry and birth-bound descendants; launch lease held until actual cleanup',
        'runner': {'returncode': process.returncode, 'timed_out': process.timed_out,
            'cancelled': process.cancelled, 'unavailable': process.unavailable,
            'resource_exhausted': process.resource_exhausted, 'output_truncated': process.output_truncated,
            'workspace_cleaned': process.workspace_cleaned, 'elapsed_ms': process.elapsed_ms,
            'max_output_bytes': bounds.max_output_bytes, 'max_workspace_bytes': bounds.max_workspace_bytes,
            'max_file_bytes': bounds.max_file_bytes, 'resident_memory_bytes': bounds.resident_memory_bytes}}
    log.write(process.stdout.encode('utf-8'))
    log.write(process.stderr.encode('utf-8'))
    log.write(b'\n' + _wire({'parent_owned_launch_lifecycle': cleanup}) + b'\n')
    log.flush()
    os.fsync(log.fileno())
    native.require(not any((process.timed_out, process.cancelled, process.unavailable,
        process.resource_exhausted, process.output_truncated)), 'fresh cold receiver native runner exceeded its bounds')
    return process.returncode, identity, cleanup


class _ColdRSSObserver(_RSSObserver):
    """Export the cold leader's real guard samples, preserving native values."""

    def bind(self, cleanup):
        lifecycle = cleanup['native_lifecycle']
        leader = lifecycle['leader']
        sample = self.samples.get((leader['pid'], leader['birth_ticks']))
        bound = cleanup['runner']['resident_memory_bytes']
        native.require(lifecycle['cleanup_confirmed'] is True and cleanup['runner']['workspace_cleaned'] is True
            and sample is not None and sample['leader'] == leader and sample['sample_count'] > 0
            and sample['tree_sample_count'] > 0 and type(bound) is int
            and 0 < sample['peak_rss_bytes'] <= bound and 0 < sample['tree_peak_rss_bytes'] <= bound,
            'cold native leader has no positive actual birth-bound guard/tree RSS observations')
        self.cold_child = {**sample, 'resident_memory_limit_bytes': bound,
            'actual_runner_returned': True, 'cleanup_confirmed': True}

    def document(self):
        native.require(hasattr(self, 'parent_record') and hasattr(self, 'cold_child') and not self.errors,
            'cold launch guard and parent monitors must drain before report publication')
        return {'schema': 'full300-paired-shared-successor-cold-launch-rss-observation@1',
            'mode': self.mode, 'owner_identity': self.parent.identity, 'parent': self.parent_record,
            'cold_child': self.cold_child, 'max_observed_leaders': 64, 'observation_errors': self.errors,
            'sampled_guard_values_returned_unchanged': True, 'kernel_memory_attestation': False}


def _parse_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            native.require(key not in value, 'duplicate campaign JSON field')
            value[key] = item
        return value
    def finite_number(value):
        number = float(value)
        native.require(math.isfinite(number), 'nonfinite campaign JSON number')
        return number
    def nonfinite(value):
        raise AssertionError('nonfinite campaign JSON: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_float=finite_number, parse_constant=nonfinite)


def _json_pin(path, maximum=1_048_576):
    raw = shared.read_absolute(Path(path), maximum)
    return _parse_json(raw), shared.pin(raw)


def _same_process(one, other):
    return type(one) is dict and type(other) is dict and all(one.get(key) == other.get(key)
        for key in ('pid', 'birth', 'parent_pid', 'group_pid', 'session_pid'))


def _native_identity_matches(identity, native_identity):
    return (identity['pid'] == native_identity['pid'] and int(identity['birth']) == native_identity['birth_ticks']
        and identity['parent_pid'] == native_identity['parent_pid']
        and identity['group_pid'] == native_identity['process_group_id']
        and identity['session_pid'] == native_identity['session_id'])


def _live_launch_parent(scheduler, request, *, exclusive_pool=False):
    native.require(type(exclusive_pool) is bool, 'explicit cold parent pool observation required')
    parent = custody._process(os.getppid())
    native.require(_same_process(parent, request['parent_identity']) and parent['state'] not in {'Z', 'X', 'x'},
        'cold receiving invoking parent identity changed')
    locator = request['orchestration']
    fields = {'lease_id', 'lane', 'cpu_slots', 'memory_mb', 'gpu_memory_mb', 'unified_memory_mb',
        'child_process_slots', 'requires_gpu', 'parent_lease_id', 'owner_pid', 'acquired_at',
        'wait_seconds', 'released', 'cancelled'}
    native.require(type(locator) is dict and set(locator) == fields
        and type(locator['lease_id']) is str and bool(locator['lease_id'])
        and type(locator['owner_pid']) is int and locator['owner_pid'] == parent['pid']
        and locator['released'] is False and locator['cancelled'] is False and locator['requires_gpu'] is False
        and locator['parent_lease_id'] is None and locator['lane'] == 'orchestration',
        'closed cold launch reservation locator required')
    for key, expected in (('cpu_slots', 1), ('memory_mb', 512), ('child_process_slots', 1),
            ('gpu_memory_mb', 0), ('unified_memory_mb', 0)):
        native.require(type(locator[key]) is int and locator[key] == expected, 'cold launch locator resources differ')
    native.require(type(locator['acquired_at']) in {int, float} and math.isfinite(locator['acquired_at'])
        and locator['acquired_at'] > 0 and type(locator['wait_seconds']) in {int, float}
        and math.isfinite(locator['wait_seconds']) and locator['wait_seconds'] >= 0,
        'finite actual cold launch lease times required')
    with scheduler._locked_state(persist=False) as state:
        held = state['leases'].get(locator['lease_id'])
        native.require(type(held) is dict and held['owner_pid'] == parent['pid']
            and held['owner_birth_marker'] == parent['birth'] and held['lane'] == locator['lane']
            and held['acquired_at'] == locator['acquired_at'] and held['cpu_slots'] == 1
            and type(held['cpu_slots']) is int and type(held['memory_mb']) is int
            and held['memory_mb'] == 512 and type(held['child_process_slots']) is int
            and held['child_process_slots'] == 1 and type(held.get('gpu_memory_mb')) is int
            and held['gpu_memory_mb'] == 0 and type(held.get('unified_memory_mb')) is int
            and held['unified_memory_mb'] == 0 and held.get('requires_gpu') is False
            and held.get('parent_lease_id') is None and not held.get('cancelled')
            and not held.get('release_requested') and held['expires_at'] > time.time(),
            'actual live parent-owned cold launch lease missing or revoked')
        if exclusive_pool:
            native.require(type(state['leases']) is dict and set(state['leases']) == {locator['lease_id']}
                and type(state['waiters']) is dict and not state['waiters'],
                'cold pool contains another reservation or waiter')
    return parent


def _cold_resources(scheduler, request):
    clean = shared.assert_clean(scheduler)
    _live_launch_parent(scheduler, request, exclusive_pool=True)
    native.require(clean['global_active_lease_count'] == 1 and clean['global_waiting_request_count'] == 0,
        'cold receive must drain its owners and retain exactly its actual parent launch lease')
    return clean


def _cold_scheduler(authority_path, request):
    """Attach under the one actual live parent-owned cold launch reservation.

    Preserve the setup helper's full authority/boundary/config/host guards; the
    current pool must contain exactly this local parent lease and no waiters.
    Host lease PIDs remain provenance outside this isolated PID namespace.
    """
    raw = shared.read_absolute(authority_path)
    value = shared.inert.parse(raw)
    fields = {"schema", "state_path", "persisted_config", "auto_renew_leases",
        "lease_ttl_seconds", "parent_host_envelope", "namespace", "scope",
        "independent_whole_host_pool", "shares_host_pid_state", "proof_authority",
        "initial_resources"}
    native.require(type(value) is dict and set(value) == fields
        and value["schema"] == "successor-dispatch-container-resource-authority@1"
        and value["state_path"] == "/results/container-resource-admission.json"
        and value["auto_renew_leases"] is True
        and all(value[name] is False for name in
            ("independent_whole_host_pool", "shares_host_pid_state", "proof_authority"))
        and value["scope"] == "local PID accounting inside held host envelope and actual CPU RAM PID cgroup",
        "exact container-local resource authority required")
    native.require(value["namespace"] == {name: os.readlink("/proc/self/ns/" + name)
        for name in ("pid", "mnt", "net")}, "resource authority belongs to another namespace")
    config = value["persisted_config"]
    native.require(type(config) is dict and config.get("proof_safety_enabled") is True
        and config.get("lane_reservations") == {}, "native proof-safe local pool required")
    for name, ceiling in (("total_cpu_slots", 12), ("total_memory_mb", 8192),
            ("total_child_process_slots", 12), ("total_unified_memory_mb", 8192)):
        native.require(type(config.get(name)) is int and 0 < config[name] <= ceiling,
            "local pool exceeds held host/cgroup ceiling: " + name)
    native.require(config.get("total_gpu_memory_mb") is None
        and config["total_unified_memory_mb"] == config["total_memory_mb"],
        "CPU-only local memory pool required")
    parent = value["parent_host_envelope"]
    native.require(type(parent) is dict and set(parent) == {"host_scheduler_state_path",
        "host_configuration_pin", "host_reservation", "container_id", "cpu_limit",
        "memory_limit_bytes", "pids_limit"}
        and type(parent["cpu_limit"]) is int and parent["cpu_limit"] == 12
        and type(parent["memory_limit_bytes"]) is int and parent["memory_limit_bytes"] == 8 * 1024**3
        and type(parent["pids_limit"]) is int and parent["pids_limit"] == 512
        and type(parent["host_scheduler_state_path"]) is str
        and parent["host_scheduler_state_path"] != value["state_path"],
        "held host envelope and actual container limits required")
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution import _root_file
    boundary_path = Path("/opt/ipfs-supervisor/container-boundary.json")
    boundary_digest = _root_file(boundary_path)
    boundary_raw = shared.read_absolute(boundary_path, 65_536)
    boundary = shared.inert.parse(boundary_raw)
    native.require(hashlib.sha256(boundary_raw).hexdigest() == boundary_digest == _root_file(boundary_path)
        and boundary["schema"] == "supervisor-container-worker-boundary@1"
        and boundary["container_id"] == parent["container_id"]
        and boundary["namespaces"] == value["namespace"],
        "root-controlled worker boundary and resource container differ")
    lease = parent["host_reservation"]
    native.require(type(lease) is dict and lease.get("released") is False
        and lease.get("cancelled") is False and lease.get("requires_gpu") is False,
        "held CPU-only parent host lease required")
    for name, expected in (("cpu_slots", 12), ("memory_mb", 8192), ("child_process_slots", 12)):
        native.require(type(lease.get(name)) is int and lease[name] == expected,
            "parent host lease does not cover local container: " + name)
    ttl = value["lease_ttl_seconds"]
    native.require(type(ttl) in (int, float) and math.isfinite(ttl) and ttl > 0,
        "finite native resource lease TTL required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as scheduling
    state_path = shared.canonical_path(Path(value["state_path"]))
    for path in (state_path, state_path.with_name(state_path.name + ".lock")):
        metadata = path.lstat()
        native.require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            and path.resolve(strict=True) == path and not path.is_symlink()
            and metadata.st_uid == os.getuid(), "existing canonical owner-controlled cold state and lock required")

    class ExistingColdScheduler(scheduling.GlobalResourceScheduler):
        def _new_state(self):
            raise scheduling.ResourceConfigurationError("cold pool creation refused")

        def _validate_state_configuration(self, state, *, allow_reconfigure=False):
            if (type(state) is not dict or state.get("schema_version") != scheduling.RESOURCE_SCHEDULER_SCHEMA_VERSION
                    or state.get("config") != config):
                raise scheduling.ResourceConfigurationError("cold pool reconfiguration refused")
            return super()._validate_state_configuration(state, allow_reconfigure=False)

    scheduler = ExistingColdScheduler(scheduling.ResourceSchedulerConfig(state_path=state_path,
        auto_renew_leases=True, **config))
    native.require(scheduler.state_path == state_path and scheduler.config.persisted_dict() == config,
        "cold scheduler detached from its exact existing container-local pool")
    native.require(scheduler.config.lease_ttl_seconds == ttl
        and scheduler.config.auto_renew_leases is True, "resource renewal authority differs")
    initial = value["initial_resources"]
    native.require(type(initial) is dict and initial.get("state_path") == value["state_path"]
        and all(type(initial.get(name)) is int and initial[name] == 0
            for name in ("active_lease_count", "waiting_request_count")),
        "fresh local pool must have no earlier process owners")
    _cold_resources(scheduler, request)
    return scheduler, value, raw



def _cold_receive(request_path, expected_sha256):
    """Reopen genuine native owners and create a fresh local one-use gate."""
    request_path = Path(request_path)
    raw = shared.read_absolute(request_path, 1_048_576)
    native.require(type(expected_sha256) is str and len(expected_sha256) == 64
        and hashlib.sha256(raw).hexdigest() == expected_sha256, 'cold request raw bytes differ')
    request = _parse_json(raw)
    native.require(type(request) is dict and set(request) == COLD_REQUEST_FIELDS
        and request['schema'] == COLD_REQUEST and request['mode'] in ('default', 'reference')
        and type(request['versions']) is dict and set(request['versions']) == {'root', 'child'}
        and type(request['pages']) is list and len(request['pages']) == PAGES
        and type(request['timeout_seconds']) in {int, float} and math.isfinite(request['timeout_seconds'])
        and 0 < request['timeout_seconds'] <= 600, 'closed bounded actual cold request required')
    output, mode = Path(request['output']), request['mode']
    native.require(output.is_absolute() and output.resolve(strict=True) == output
        and request_path == output / (mode + '-cold-receive-request.json')
        and Path(request['result']) == output / (mode + '-cold-receive-result.json')
        and Path(request['rss_result']) == output / (mode + '-cold-receive-rss.json')
        and not Path(request['result']).exists() and not Path(request['rss_result']).exists(),
        'exact fresh cold request/result/RSS paths required')
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
    scheduler, authority, authority_raw = _cold_scheduler(Path(request['authority']), request)
    native.require(hashlib.sha256(authority_raw).hexdigest() == request['authority_sha256'],
        'cold actual container authority raw bytes changed')
    parent = _live_launch_parent(scheduler, request)
    selected, generation_pin = _json_pin(output / 'generation-inputs.json')
    native.require(generation_pin['sha256'] == request['generation_inputs_sha256'],
        'cold selected producer inventory raw bytes changed')
    native._require_pins(output, selected)
    scan, records = _protocol(mode)
    index = registry = connection = None
    started, deadline = time.monotonic(), time.monotonic() + request['timeout_seconds']
    report = {'schema': COLD_RESULT, 'qualified': False, 'full_campaign_qualified': False,
        'pid': os.getpid(), 'process_identity': custody._process(os.getpid()), 'parent_identity': parent,
        'mode': mode, 'request_sha256': expected_sha256, 'authority_sha256': request['authority_sha256'],
        'orchestration_lease_id': request['orchestration']['lease_id'],
        'generation_inputs_sha256': request['generation_inputs_sha256'],
        'selection_cid': request['selection_cid'], 'source_delta_cid': request['source_delta_cid'],
        'root_cid': request['root_cid'], 'completion_cid': request['completion_cid'], 'pages': [],
        'versions': request['versions'], 'expected_head': request['expected_head'],
        'checkpoint_states_before': None, 'checkpoint_states_after': None,
        'variant_heads_before': None, 'variant_heads_after': None,
        'registry_owner_before': None, 'registry_owner_after': None, 'paired_receiver_phase_record': None,
        'post_setup_fit_attempt_count': 0, 'new_fitting_epochs': 0, 'receive_forward_attempt_count': 0,
        'receiving_extra_numerical_attempt_count': 0, 'receiving_mutation_attempt_count': 0,
        'no_install_guards': None, 'rss_observation_file': mode + '-cold-receive-rss.json',
        'rss_observation_pin': None, 'selected_producers_unchanged': False, 'native_owners_closed': False,
        'resources_after': None, 'cleanup_errors': [], 'proof_authority': False,
        'cuda_qualified': False, '384d_qualified': False, '768d_qualified': False, '4096d_qualified': False}
    monitor, phase_report = None, None
    def options():
        remaining = deadline - time.monotonic()
        native.require(remaining > 0, 'cold receiving deadline exceeded')
        return {'scheduler': scheduler, 'memory_mb': 1024, 'timeout_seconds': min(580, remaining),
            'admission_timeout_seconds': min(30, remaining)}
    try:
        monitor = _ParentRSSMonitor()
        index, registry, connection = native._open(output)
        before = _states(registry, request['versions'])
        report['checkpoint_states_before'] = before
        report['variant_heads_before'], report['registry_owner_before'] = _variant_heads(registry, request['versions']), _registry_owner(registry)
        native.require(before == request['checkpoint_states']
            and index.current(REPOSITORY_ID).to_dict() == request['expected_head'], 'cold source/checkpoint inputs differ')
        root = resume.load_codebase_scan_resume_root(index.artifacts, request['root_cid'])
        selection = successor.load_codebase_successor_scan(index.artifacts, request['selection_cid'])
        source_delta = delta.load_codebase_source_delta(index.artifacts, request['source_delta_cid'])
        completion = scan.load_codebase_scan_chunked_completion(index.artifacts, request['completion_cid'])
        native.require(selection.to_dict()['root_cid'] == root.artifact_cid
            and selection.to_dict()['source_delta_cid'] == source_delta.artifact_cid
            and selection.to_dict()['previous_model']['version_id'] == request['versions']['root']
            and selection.to_dict()['model']['version_id'] == request['versions']['child']
            and source_delta.to_dict()['current_head'] == request['expected_head']
            and root.to_dict()['head'] == request['expected_head']
            and root.to_dict()['optimized'] is (mode == 'default')
            and len(root.to_dict()['members']) == ENTRIES
            and root.to_dict()['limits']['max_inventory_entries'] == 512
            and root.to_dict()['limits']['page_entries'] == PAGE_ENTRIES,
            'cold exact selected direct-child/delta/root/head profile differs')
        report['pages'] = [records.page_descriptor(scan.load_codebase_scan_chunked_page(index.artifacts, value['page_cid']))
            for value in completion.to_dict()['pages']]
        native.require(report['pages'] == request['pages'] == completion.to_dict()['pages'],
            'cold full saved page descriptor population differs')
        for number, descriptor in enumerate(report['pages'], 1):
            page = scan.load_codebase_scan_chunked_page(index.artifacts, descriptor['page_cid'])
            _verify_page_exports(output, mode, number, page, index)
        def late_callback(actual_selection, actual_completion):
            native.require(actual_selection is selection and actual_completion is completion
                and index.current(REPOSITORY_ID).to_dict() == request['expected_head']
                and _states(registry, request['versions']) == before
                and _variant_heads(registry, request['versions']) == report['variant_heads_before'],
                'late cold callback source/weights/Adam/variant heads changed')
            _live_launch_parent(scheduler, request)
        with custody._no_install_guards(report):
            phase_report = _receive_selected_pair(mode, 'cold', selection, completion, index, output / 'repository',
                root=root, registry=registry, versions=request['versions'], frozen_states=before,
                state_reader=_states, options=options(), late_callback=late_callback,
                no_forward_guard=_no_forward, no_fit_guard=shared.no_fit,
                rss_monitor_factory=_ParentRSSMonitor, process_identity_reader=custody._process, report=report)
        report['paired_receiver_phase_record'] = phase_report
        report['checkpoint_states_after'] = _states(registry, request['versions'])
        report['variant_heads_after'], report['registry_owner_after'] = _variant_heads(registry, request['versions']), _registry_owner(registry)
        native.require(report['checkpoint_states_after'] == request['checkpoint_states']
            and report['variant_heads_after'] == report['variant_heads_before']
            and report['registry_owner_after'] == report['registry_owner_before'],
            'cold receiving changed weights/Adam/report/variant heads or its owner generation')
        _live_launch_parent(scheduler, request, exclusive_pool=True)
        native._require_pins(output, selected)
        native.require(shared.pin(shared.read_absolute(output / 'generation-inputs.json', 1_048_576)) == generation_pin,
            'cold producer inventory changed after native receiving')
        report['selected_producers_unchanged'] = True
        absent = {'intent_database_absent': not (output / 'private/intent.duckdb').exists(),
            'worker_launch_directory_absent': not (output / 'private/launch').exists(),
            'worker_worktrees_absent': not (output / 'repository/.git/worktrees').exists()}
        native.require(all(absent.values()) and all(value == 0 for value in report['no_install_guards']['counters'].values()),
            'cold receiving attempted worker/intent installation')
        report['no_install_guards'].update(absent)
        report['qualified'] = True
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary = sys.exc_info()[0] is not None
        cleanup_error = None
        partial = report.pop('paired_receiver_phase_records', [])
        if report['paired_receiver_phase_record'] is None and partial:
            report['paired_receiver_phase_record'] = partial[-1]
        for name, action in (('close_native_owners', lambda: native._close(registry, connection)),
                ('observe_owned_resources', lambda: _cold_resources(scheduler, request))):
            try:
                value = action()
                if name == 'close_native_owners':
                    report['native_owners_closed'] = True
                else:
                    report['resources_after'] = value
            except BaseException as error:
                report['qualified'] = False
                report['cleanup_errors'].append({'stage': name, 'error_type': type(error).__name__, 'error': str(error)})
                cleanup_error = cleanup_error or error
        try:
            native.require(monitor is not None, 'actual cold process RSS monitor required')
            whole = monitor.finish()
            native.require(_native_identity_matches(report['process_identity'], whole['identity'])
                and 0 < whole['peak_rss_bytes'] <= 8 * 1024**3, 'cold real process RSS identity/bound differs')
            memory = {'schema': COLD_RSS, 'mode': mode, 'process_identity': report['process_identity'],
                'request_sha256': expected_sha256, 'whole_process': whole,
                'phase': None if report['paired_receiver_phase_record'] is None else report['paired_receiver_phase_record']['rss_observation'],
                'sampled_memory_observation_only': True, 'kernel_memory_attestation': False}
            base._write(Path(request['rss_result']), memory)
            report['rss_observation_pin'] = shared.pin(shared.read_absolute(Path(request['rss_result']), 1_048_576))
            if report['paired_receiver_phase_record'] is not None:
                base._write(output / (mode + '-cold-receive-phase.json'), report['paired_receiver_phase_record'])
        except BaseException as error:
            report['qualified'] = False
            report['cleanup_errors'].append({'stage':'finish_actual_rss', 'error_type':type(error).__name__, 'error':str(error)})
            cleanup_error = cleanup_error or error
        base._write(Path(request['result']), report)
        if cleanup_error is not None and not primary:
            raise cleanup_error
    return report


def _join_cold_receive(output, mode, request, request_pin, identity, cleanup, launch_memory,
        selection, root, completion, versions, frozen_states, frozen_heads):
    result_path, rss_path = output / (mode + '-cold-receive-result.json'), output / (mode + '-cold-receive-rss.json')
    result, result_pin = _json_pin(result_path)
    memory, memory_pin = _json_pin(rss_path)
    phase_value, phase_pin = _json_pin(output / (mode + '-cold-receive-phase.json'))
    native.require(type(result) is dict and set(result) == COLD_RESULT_FIELDS
        and result['schema'] == COLD_RESULT and result['qualified'] is True
        and result['full_campaign_qualified'] is False and result['mode'] == mode
        and result['pid'] == identity['pid'] and _same_process(result['process_identity'], identity)
        and _same_process(result['parent_identity'], request['parent_identity'])
        and _native_identity_matches(identity, cleanup['native_lifecycle']['leader'])
        and result['request_sha256'] == request_pin['sha256']
        and result['authority_sha256'] == request['authority_sha256']
        and result['generation_inputs_sha256'] == request['generation_inputs_sha256']
        and result['orchestration_lease_id'] == request['orchestration']['lease_id'],
        'actual cold result/request/live birth identity joins differ')
    expected = {'selection_cid':selection.artifact_cid, 'source_delta_cid':selection.to_dict()['source_delta_cid'],
        'root_cid':root.artifact_cid, 'completion_cid':completion.artifact_cid,
        'pages':completion.to_dict()['pages'], 'versions':versions, 'expected_head':root.to_dict()['head']}
    native.require(all(result[name] == value == request[name] for name,value in expected.items())
        and result['checkpoint_states_before'] == result['checkpoint_states_after'] == request['checkpoint_states'] == frozen_states
        and result['variant_heads_before'] == result['variant_heads_after'] == frozen_heads
        and result['registry_owner_before'] == result['registry_owner_after']
        and result['registry_owner_before']['pid'] == identity['pid'],
        'cold actual selected artifacts/pages/weights/Adam/variant heads/owner generation differ')
    counters = ('post_setup_fit_attempt_count', 'new_fitting_epochs', 'receive_forward_attempt_count',
        'receiving_extra_numerical_attempt_count', 'receiving_mutation_attempt_count')
    native.require(all(type(result[name]) is int and result[name] == 0 for name in counters)
        and result['native_owners_closed'] is True and result['selected_producers_unchanged'] is True
        and result['cleanup_errors'] == []
        and all(value == 0 for value in result['no_install_guards']['counters'].values())
        and all(result['no_install_guards'][name] is True for name in
            ('intent_database_absent','worker_launch_directory_absent','worker_worktrees_absent')),
        'cold receiving numerical/mutation/install/owner cleanup accounting differs')
    native.require(type(memory) is dict and set(memory) == {'schema','mode','process_identity',
            'request_sha256','whole_process','phase','sampled_memory_observation_only','kernel_memory_attestation'}
        and memory['schema'] == COLD_RSS and memory['mode'] == mode
        and _same_process(memory['process_identity'], identity) and memory['request_sha256'] == request_pin['sha256']
        and result['rss_observation_file'] == rss_path.name and result['rss_observation_pin'] == memory_pin
        and memory['sampled_memory_observation_only'] is True and memory['kernel_memory_attestation'] is False,
        'cold actual RSS file/raw SHA/result join differs')
    phase_record = result['paired_receiver_phase_record']
    native.require(phase_record == phase_value and phase_record['schema'] == RECEIVE_PHASE_SCHEMA
        and phase_record['profile'] == RECEIVER_PROFILE and phase_record['mode'] == mode and phase_record['stage'] == 'cold'
        and phase_record['accepted'] is True and phase_record['native_scope_exit_success'] is True
        and phase_record['callback_invocations'] == phase_record['successful_close_calls'] == 1
        and phase_record['full_campaign_qualified'] is False
        and _same_process(phase_record['invoking_process_identity'], identity)
        and memory['phase'] == phase_record['rss_observation']
        and phase_record['checkpoint_states_before'] == phase_record['checkpoint_states_after'] == frozen_states
        and phase_record['variant_heads_before'] == phase_record['variant_heads_after'] == frozen_heads
        and phase_record['registry_owner_before'] == phase_record['registry_owner_after'] == result['registry_owner_before']
        and all(phase_record[key] == expected[key] for key in ('selection_cid','root_cid','completion_cid','source_delta_cid')),
        'cold actual raw phase/request/artifact/identity/state joins differ')
    for observation in (memory['whole_process'], memory['phase']):
        native.require(_native_identity_matches(identity, observation['identity'])
            and observation['monitor_joined'] is True and observation['errors'] == []
            and type(observation['sample_count']) is int and observation['sample_count'] > 0
            and type(observation['peak_rss_bytes']) is int and 0 < observation['peak_rss_bytes'] <= 8 * 1024**3,
            'cold positive actual process/phase RSS evidence differs')
    guard = launch_memory['cold_child']
    native.require(_native_identity_matches(identity, guard['leader']) and guard['sample_count'] > 0
        and guard['tree_sample_count'] > 0 and guard['cleanup_confirmed'] is True
        and guard['actual_runner_returned'] is True and launch_memory['observation_errors'] == []
        and guard['resident_memory_limit_bytes'] == 8 * 1024**3
        and 0 < guard['peak_rss_bytes'] <= guard['resident_memory_limit_bytes']
        and 0 < guard['tree_peak_rss_bytes'] <= guard['resident_memory_limit_bytes'],
        'cold actual parent guard/tree samples do not join its drained leader')
    native.require(shared.pin(shared.read_absolute(result_path, 1_048_576)) == result_pin
        and shared.pin(shared.read_absolute(rss_path, 1_048_576)) == memory_pin
        and shared.pin(shared.read_absolute(output / (mode + '-cold-receive-phase.json'), 1_048_576)) == phase_pin,
        'cold result/RSS/phase raw bytes changed during parent joins')
    return result, {'result':result_pin, 'rss':memory_pin, 'phase':phase_pin}


def _receiving_refusal_controls(selection, root, completion, index, repository, registry, versions, frozen, options, report):
    """Two real cheap full300 refusals, without changing durable scan bytes."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_shared_successor_receiving as receiving
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    _, records = _protocol('default')
    value = completion.to_dict()
    value['root_cid'] = cid_for_structured({'schema':'full300-paired-receiving-foreign-root-control@1'})
    raw = records._wire(value)
    wrong_root = records.CodebaseScanChunkedCompletion.from_dict(cid_for_bytes(raw), value)
    event = threading.Event()
    event.set()
    for name, selected_completion, changes, refusal in (
            ('pre_cancelled_full300_pair',completion,{'cancel_event':event},LeaseCancelledError),
            ('wrong_root_full300_completion',wrong_root,{},resume.CodebaseScanResumeError)):
        entered = False
        before = _states(registry, versions)
        try:
            with _receiving_guards(report,no_forward_guard=_no_forward,no_fit_guard=shared.no_fit):
                with receiving._paired_current_codebase_shared_successor_completion(selection, selected_completion,
                        index, repository, root=root, registry=registry, **options(), **changes):
                    entered = True
        except refusal as error:
            row = {'schema':'full300-paired-shared-successor-refusal-control@1', 'name':name,
                'expected':'refuse', 'observed':'refuse', 'native_gate_entered':entered,
                'error_type':type(error).__name__, 'reason':str(error)[:2048],
                'selection_cid':selection.artifact_cid, 'root_cid':root.artifact_cid,
                'completion_cid':selected_completion.artifact_cid, 'durable_scan_bytes_mutated':False,
                'checkpoint_states_before':before, 'checkpoint_states_after':_states(registry, versions),
                'proof_authority':False}
        else:
            raise AssertionError('full300 actual paired refusal control accepted: ' + name)
        native.require(not entered and row['checkpoint_states_before'] == row['checkpoint_states_after'] == frozen
            and all(report[name] == 0 for name in ('post_setup_fit_attempt_count','receive_forward_attempt_count',
                'receiving_extra_numerical_attempt_count','receiving_mutation_attempt_count')),
            'actual full300 refusal changed state or attempted numerical/mutation work')
        report['controls'].append(row)


def _require_receiving_exports(output, report):
    for row in report['paired_receiver_phase_files']:
        native.require(shared.pin(shared.read_absolute(output / row['file'],1_048_576)) == row['pin'],
            'accepted warm/cold phase raw bytes changed before final qualification')
    for row in report['fresh_cold_receiver_processes']:
        mode = row['mode']
        native.require(shared.pin(shared.read_absolute(output / (mode + '-cold-receive-request.json'),1_048_576))
            == row['request_pin'] and shared.pin(shared.read_absolute(output / row['launch_rss_file'],1_048_576))
            == row['launch_rss_pin'], 'actual cold request or native guard RSS raw bytes changed')
        for field,name in (('result',mode + '-cold-receive-result.json'),('rss',mode + '-cold-receive-rss.json'),
                ('phase',mode + '-cold-receive-phase.json')):
            native.require(shared.pin(shared.read_absolute(output / name,1_048_576)) == row['raw_output_pins'][field],
                'actual accepted cold result/RSS/phase raw bytes changed')


def _finalize_campaign(output, report, *, registry, connection, scheduler, selected_producers, started, primary_error_active):
    """Close actual owners/resources and emit exactly one exclusive result."""
    first_cleanup_error = None
    def failed(stage, error):
        nonlocal first_cleanup_error
        first_cleanup_error = first_cleanup_error or error
        report['qualified'] = False
        report.setdefault('cleanup_errors', []).append({'stage':stage,
            'error_type':type(error).__name__, 'error':str(error)})
    try:
        native._close(registry, connection)
    except BaseException as error:
        failed('close_native_owners', error)
    report['recorded_seconds'] = time.monotonic() - started
    try:
        report['final_resources'] = shared.assert_clean(scheduler)
    except BaseException as error:
        report['final_resource_cleanup_verified'] = False
        failed('verify_named_resource_cleanup', error)
    else:
        report['final_resource_cleanup_verified'] = True
    if report['qualified'] is True and not primary_error_active:
        try:
            _require_receiving_exports(output, report)
            native._require_pins(output, selected_producers)
        except BaseException as error:
            report['selected_producers_unchanged'] = False
            failed('close_selected_source_and_receiving_output_pins', error)
    report['full_campaign_qualified'] = (report['qualified'] is True
        and report['final_resource_cleanup_verified'] is True and not primary_error_active)
    try:
        base._write(Path(output) / 'result.json', report)
    except BaseException as error:
        report['full_campaign_qualified'] = False
        failed('persist_result_report', error)
    if first_cleanup_error is not None and not primary_error_active:
        raise first_cleanup_error


def _resume(request_path, expected_sha256):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    request_path = Path(request_path)
    raw = shared.read_absolute(request_path, 1_048_576)
    native.require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'chunk resume request bytes differ')
    request = json.loads(raw)
    native.require(set(request) == {'schema', 'output', 'mode', 'root_cid', 'cursor', 'versions',
        'checkpoint_states', 'expected_head', 'authority', 'authority_sha256', 'orchestration',
        'parent_identity', 'result', 'max_pages', 'timeout_seconds', 'selection_cid',
        'source_delta_cid'}, 'closed paired-successor continuation request required')
    native.require(request['schema'] == RESUME_REQUEST and request['mode'] in MODES
        and type(request['max_pages']) is int and request['max_pages'] == RESUME_PAGES
        and set(request['versions']) == {'root', 'child'}
        and type(request['timeout_seconds']) in {int, float} and math.isfinite(request['timeout_seconds'])
        and 0 < request['timeout_seconds'] <= 1800, 'bounded exact chunk continuation required')
    output, mode = Path(request['output']), request['mode']
    scan, records = _protocol(mode)
    native.require(output.is_absolute() and output.resolve(strict=True) == output
        and request_path == output / (mode + '-resume-request.json')
        and Path(request['result']) == output / (mode + '-resume-result.json'), 'exact chunk continuation paths required')
    authority_raw = shared.read_absolute(Path(request['authority']), 1_048_576)
    native.require(hashlib.sha256(authority_raw).hexdigest() == request['authority_sha256'], 'chunk resource authority changed')
    authority = json.loads(authority_raw)
    native.require(authority['namespace'] == {name: os.readlink('/proc/self/ns/' + name)
        for name in ('pid', 'mnt', 'net')}, 'chunk continuation namespace differs')
    scheduler = shared.shared_scheduler(authority['state_path'], authority['persisted_config'])
    parent = custody._process(os.getppid())
    native.require(parent is not None and all(parent[key] == request['parent_identity'][key]
        for key in ('pid', 'birth', 'parent_pid', 'group_pid', 'session_pid'))
        and parent['state'] not in {'Z', 'X', 'x'}, 'chunk continuation parent changed')
    with scheduler._locked_state(persist=False) as state:
        held = state['leases'].get(request['orchestration']['lease_id'])
        native.require(held is not None and held['owner_pid'] == os.getppid()
            and held['owner_birth_marker'] == parent['birth'] and held['cpu_slots'] == 1
            and held['memory_mb'] == 512 and held['child_process_slots'] == 1
            and not held.get('cancelled') and not held.get('release_requested')
            and held['expires_at'] > time.time(), 'chunk continuation launch reservation missing')
    selected = json.loads(shared.read_absolute(output / 'generation-inputs.json', 1_048_576))
    native._require_pins(output, selected)
    deadline = time.monotonic() + request['timeout_seconds']
    index = registry = connection = None
    report = {'schema': RESUME_RESULT, 'qualified': False, 'pid': os.getpid(),
        'process_identity': custody._process(os.getpid()), 'mode': mode,
        'request_sha256': expected_sha256, 'pages': [], 'selection_cid': request['selection_cid'],
        'source_delta_cid': request['source_delta_cid'], 'post_setup_fit_attempt_count': 0,
        'new_fitting_epochs': 0, 'receive_forward_attempt_count': 0, 'proof_authority': False}

    def options():
        remaining = deadline - time.monotonic()
        native.require(remaining > 0, 'chunk continuation deadline exceeded')
        return {'scheduler': scheduler, 'memory_mb': 1024, 'timeout_seconds': min(400, remaining),
                'admission_timeout_seconds': min(30, remaining)}

    observer = _RSSObserver(output, mode)
    try:
        index, registry, connection = native._open(output)
        native.require(index.current(REPOSITORY_ID).to_dict() == request['expected_head']
            and _states(registry, request['versions']) == request['checkpoint_states'], 'chunk resume source or state changed')
        root = resume.load_codebase_scan_resume_root(index.artifacts, request['root_cid'])
        native.require(len(root.to_dict()['members']) == ENTRIES
            and root.to_dict()['limits']['page_entries'] == PAGE_ENTRIES
            and root.to_dict()['model']['version_id'] == request['versions']['child'], 'chunk resume root differs')
        cursor = records.CodebaseScanChunkedCursor.from_dict(request['cursor'])
        native.require(cursor.next_offset == PARENT_PAGES * PAGE_ENTRIES, 'fresh chunk cursor is not page boundary128')
        with shared.no_fit(report), observer:
            native.require((request['selection_cid'] is None) is (mode == 'legacy_default'),
                'exact conditional selected successor binding required')
            if mode != 'legacy_default':
                from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
                selection = successor.load_codebase_successor_scan(index.artifacts, request['selection_cid'])
                native.require(selection.to_dict()['root_cid'] == root.artifact_cid
                    and selection.to_dict()['source_delta_cid'] == request['source_delta_cid']
                    and selection.to_dict()['previous_model']['version_id'] == request['versions']['root']
                    and selection.to_dict()['model']['version_id'] == request['versions']['child'],
                    'fresh selected successor/root/delta/model lineage differs')
                with _no_forward(report):
                    native.require(successor.validate_current_codebase_successor_scan(selection,
                        index, output / 'repository', registry=registry, **options()) is selection,
                        'fresh selected successor native receiving changed selection')
            prefix = scan.load_codebase_scan_chunked_page(index.artifacts, cursor.previous_page_cid)
            with _no_forward(report):
                native.require(scan.validate_current_codebase_chunked_scan_page(prefix, index, output / 'repository',
                    root=root, registry=registry, **options()) is prefix, 'cold prefix receiving changed selection')
            for number in range(PARENT_PAGES + 1, PAGES + 1):
                page = scan.scan_current_codebase_page_chunked(index, output / 'repository', root=root,
                    registry=registry, cursor=cursor, **options())
                report['pages'].append(_export_page(output, mode, number, page, index))
                cursor = records.CodebaseScanChunkedCursor.from_page(root, page)
            native.require(cursor is None and len(report['pages']) == RESUME_PAGES, 'fresh chunk continuation incomplete')
        report['checkpoint_states_after'] = _states(registry, request['versions'])
        native.require(report['checkpoint_states_after'] == request['checkpoint_states'], 'fresh chunks altered weights or Adam')
        native._require_pins(output, selected)
        memory = observer.document()
        base._write(output / (mode + '-resume-rss.json'), memory)
        report['rss_observation_file'] = mode + '-resume-rss.json'
        report['rss_numerical_children'] = len(memory['numerical_children'])
        report['qualified'] = True
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary = sys.exc_info()[0] is not None
        cleanup_error = None
        for name, action in (('close_native_owners', lambda: native._close(registry, connection)),
                ('observe_owned_resources', lambda: shared.assert_clean(scheduler))):
            try:
                value = action()
                if name == 'observe_owned_resources':
                    report['resources_after'] = value
            except BaseException as error:
                report['qualified'] = False
                report.setdefault('cleanup_errors', []).append({'stage': name,
                    'error_type': type(error).__name__, 'error': str(error)})
                cleanup_error = cleanup_error or error
        base._write(Path(request['result']), report)
        if cleanup_error is not None and not primary:
            raise cleanup_error
    return report


def _mapped_rows(index, page, mode):
    scan, _ = _protocol(mode)
    mapped = {}
    for descriptor in page.to_dict()['chunks']:
        chunk = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid']).to_dict()
        for offset, member_index in enumerate(chunk['member_indices']):
            native.require(member_index not in mapped, 'duplicate equal-work member mapping')
            mapped[member_index] = (chunk['source_digests'][offset], chunk['target_sha256s'][offset],
                                    chunk['inference']['rows'][offset])
        del chunk
    return mapped


def _compare_pages(index, first_pages, reference_pages, first_mode, second_mode, *, exact):
    """Compare by source/member, independently of numerical chunk partitions."""
    native.require(len(first_pages) == len(reference_pages) == PAGES, 'equal-work page population differs')
    totals = {'rows': 0, 'projection_records': 0, 'pages': 0, 'first_chunks': 0, 'second_chunks': 0}
    maximum, relative_maximum, values_exact = 0.0, 0.0, True
    tolerance = 0.0 if exact else 1e-12
    for first, reference in zip(first_pages, reference_pages):
        f, r = first.to_dict(), reference.to_dict()
        normalize = lambda entries: [{key: value for key, value in entry.items()
            if key not in {'chunk_ordinal', 'row_index'}} for entry in entries]
        native.require(shared.inert.observation_same(normalize(f['entries']), normalize(r['entries']))
            and shared.inert.observation_same({key: value for key, value in f['coverage'].items() if key != 'chunks'},
                {key: value for key, value in r['coverage'].items() if key != 'chunks'}),
            'equal-work source entries or non-partition coverage differ')
        left, right = _mapped_rows(index, first, first_mode), _mapped_rows(index, reference, second_mode)
        native.require(left.keys() == right.keys(), 'equal-work inferred member population differs')
        for member_index in left:
            source, target, row = left[member_index]
            other_source, other_target, other = right[member_index]
            native.require(source == other_source and target == other_target
                and row['source_digest'] == other['source_digest'], 'equal-work source/target identity differs')
            native.require(row['reconstructed_projection_features'].keys()
                == other['reconstructed_projection_features'].keys(), 'equal-work projection identities differ')
            arrays = [(row['latent'], other['latent'])] + [
                (array, other['reconstructed_projection_features'][name])
                for name, array in row['reconstructed_projection_features'].items()]
            for a, b in arrays:
                native.require(len(a) == len(b), 'equal-work numerical shape differs')
                for av, bv in zip(a, b):
                    native.require(type(av) is float and type(bv) is float
                        and math.isfinite(av) and math.isfinite(bv), 'equal-work finite float values required')
                    maximum = max(maximum, abs(av - bv))
                    denominator = max(abs(av), abs(bv))
                    relative_maximum = max(relative_maximum, abs(av - bv) / denominator if denominator else 0.0)
                    values_exact = values_exact and av == bv
                    native.require((av == bv) if exact else math.isclose(av, bv,
                        rel_tol=tolerance, abs_tol=tolerance), 'equal-work numerical values exceed tolerance')
        totals['rows'] += len(left)
        totals['projection_records'] += f['coverage']['projection_records']
        totals['first_chunks'] += len(f['chunks'])
        totals['second_chunks'] += len(r['chunks'])
        totals['pages'] += 1
        del left, right, f, r
    native.require(totals['rows'] == 294 and totals['projection_records'] == 588,
                   'equal-work full300 numerical totals differ')
    return {**totals, 'first_mode': first_mode, 'second_mode': second_mode, 'inventory_entries': ENTRIES,
        'per_source_target_identity_exact': True, 'entries_and_coverage_exact': True,
        'per_source_numerical_values_exact': values_exact, 'per_source_numerical_values_within_tolerance': True,
        'max_absolute_error': maximum, 'max_relative_error': relative_maximum,
        'relative_tolerance': tolerance, 'absolute_tolerance': tolerance,
        'throughput_qualified': False}


def _measurements(index, pages, mode):
    scan, _ = _protocol(mode)
    result = {'schema': 'full300-shared-actual-mode-measurements@1', 'mode': mode,
        'requests': 0, 'input_bytes': 0, 'output_bytes': 0, 'worker_elapsed_ms': 0,
        'worker_counters': {}, 'transport_bytes': {}, 'scope': 'actual recorded requests in this frozen fixture'}
    for page in pages:
        for descriptor in page.to_dict()['chunks']:
            c = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid']).to_dict()
            result['requests'] += 1
            for key in ('input_bytes', 'output_bytes'):
                result[key] += c['worker_receipt'][key]
            result['worker_elapsed_ms'] += c['worker_receipt']['elapsed_ms']
            for key, value in c['worker_counters'].items():
                if key == 'max_retained_restored_targets':
                    result['worker_counters'][key] = max(result['worker_counters'].get(key, 0), value)
                else:
                    result['worker_counters'][key] = result['worker_counters'].get(key, 0) + value
            for key, value in c.get('transport_bytes', {}).items():
                if key == 'maximum_restored_targets':
                    result['transport_bytes'][key] = max(result['transport_bytes'].get(key, 0), value)
                else:
                    result['transport_bytes'][key] = result['transport_bytes'].get(key, 0) + value
    return result


def _join_rss(index, pages, mode, observations, output, root):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    scan, _ = _protocol(mode)
    children = [row for observation in observations for row in observation['numerical_children']]
    by_leader = {(row['leader']['pid'], row['leader']['birth_ticks']): row for row in children}
    native.require(len(children) == len(by_leader), 'duplicate numerical RSS leader identity')
    accepted = set()
    for page in pages:
        for descriptor in page.to_dict()['chunks']:
            c = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid']).to_dict()
            leader = c['lifecycle_receipt']['leader']
            key = (leader['pid'], leader['birth_ticks'])
            sample = by_leader.get(key)
            native.require(sample is not None and sample['leader'] == leader
                and sample['input_bytes'] == c['worker_receipt']['input_bytes']
                and sample['input_sha256'] == c['worker_receipt']['input_sha256']
                and sample['resident_memory_limit_bytes'] == c['worker_receipt']['limits']['resident_memory_bytes'],
                'actual numerical RSS, input or bound differs from native chunk receipt')
            native.require(key not in accepted, 'chunk reuses a numerical RSS leader')
            response = {'schema': 'codebase-inventory-resume-worker@1' if mode == 'legacy_default'
                else 'codebase-inventory-resume-shared-worker@1', 'optimized': c['optimized'],
                'inference': c['inference'], 'counters': c['worker_counters'], 'worker': c['worker_profile']}
            response_raw = resume._wire(response)
            native.require(len(response_raw) == c['worker_receipt']['output_bytes']
                and hashlib.sha256(response_raw).hexdigest() == c['worker_receipt']['output_sha256'],
                'actual numerical stdout bytes differ from saved chunk response')
            if mode == 'legacy_default':
                native.require(sample['input_file'] is None, 'legacy full requests must not be retained as shared inputs')
            else:
                name = mode + '-worker-input-' + str(c['member_indices'][0]).zfill(4) + '.json'
                native.require(sample['input_file'] == name, 'actual shared stdin locator differs')
                raw = shared.read_absolute(output / name, 32 * 1024 * 1024)
                native.require(len(raw) == sample['input_bytes']
                    and hashlib.sha256(raw).hexdigest() == sample['input_sha256'], 'actual shared stdin bytes changed')
                value = json.loads(raw)
                native.require(value['schema'] == response['schema'] and value['root_cid'] == root.artifact_cid
                    and resume._wire(value['root']) == resume._wire(root.to_dict())
                    and value['optimized'] is c['optimized'] and value['member_indices'] == c['member_indices']
                    and value['shared_inventory']['shared_sha256'] == c['shared_inventory_sha256']
                    and len(value['targets']) == len(c['member_indices'])
                    and [target['target_sha256'] for target in value['targets']] == c['target_sha256s']
                    and [target['source_digest'] for target in value['targets']] == c['source_digests']
                    and features.digest(value['state']) == c['model_state_sha256']
                    and features.digest(value['feature_space']) == c['inference']['feature_space_sha256'],
                    'actual shared stdin root/member/model/source/target joins differ')
                del raw, value
            accepted.add(key)
    native.require(accepted == by_leader.keys(), 'RSS numerical leader population differs from accepted chunks')
    return {'numerical_children': len(children), 'peak_rss_bytes': max(row['peak_rss_bytes'] for row in children),
        'tree_peak_rss_bytes': max(row['tree_peak_rss_bytes'] for row in children),
        'sample_count': sum(row['sample_count'] for row in children),
        'tree_sample_count': sum(row['tree_sample_count'] for row in children),
        'parent_peak_rss_bytes': max(observation['parent']['peak_rss_bytes'] for observation in observations),
        'sampled_population_exact': True, 'kernel_memory_attestation': False}


def run(output, *, overall_seconds=5950, host_configuration='/results/container-resource-authority.json'):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunks as records
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    native.require(type(overall_seconds) is int and 1200 <= overall_seconds <= 6000,
                   'bounded full300 chunk qualification deadline required')
    output = Path(output).absolute()
    native.require(not output.exists() and output.parent.resolve(strict=True) == output.parent, 'fresh exact chunk output required')
    output.mkdir()
    scheduler, authority, authority_raw = deployment._container_scheduler(Path(host_configuration))
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {'schema': SCHEMA, 'qualified': False, 'pid': os.getpid(),
        'process_identity': custody._process(os.getpid()), 'phases': [], 'controls': [],
        'scope': 'fresh300_cpu_structural8d_294_actual_rows_three_scan_modes_and_four_paired_shared_successor_receives_without_signed_dispatch',
        'inventory_entries': ENTRIES, 'page_entries': PAGE_ENTRIES, 'new_fitting_epochs': 0,
        'new_legacy_default_scan_pages': 0, 'new_default_scan_pages': 0, 'new_reference_scan_pages': 0, 'fresh_resume_processes': [],
        'post_setup_fit_attempt_count': 0, 'receive_forward_attempt_count': 0, 'provider_calls': 0,
        'receiving_extra_numerical_attempt_count': 0, 'receiving_mutation_attempt_count': 0,
        'fresh_cold_receiver_processes': [], 'paired_receiver_phase_records': [],
        'registry_owner_observations': [], 'paired_receiver_phase_files': [],
        'full_campaign_qualified': False, 'warm_paired_receiver_qualified': False,
        'cold_paired_receiver_qualified': False, 'paired_shared_successor_receiving_qualified': False,
        'full300_refusal_controls_qualified': False, 'full300_late_mutation_controls_qualified': False,
        'full300_midflight_cancellation_controls_qualified': False,
        'worker_launched': False, 'task_installation_qualified': False, 'native_worker_qualified': False,
        'worker_launch_flag_scope': 'supervisor task worker; numerical child launches are recorded separately',
        'actual_numerical_requests_by_mode': {}, 'actual_mode_measurements': {}, 'actual_rss_by_mode': {},
        'rss_observation_files': {}, 'retained_shared_worker_inputs': True, 'throughput_qualified': False,
        'worker_profile_integrated': False, 'signed_admission_qualified': False,
        'signed_chunked_context_integration_qualified': False, 'successor_dispatch_qualified': False,
        'signed_preview_companion_qualified': False, 'publication_qualified': False,
        'proof_authority': False, 'source_execution_attested': False, 'scan_execution_attested': False,
        'cuda_qualified': False, '384d_qualified': False, '768d_qualified': False, '4096d_qualified': False,
        'semantic_reproof_join_qualified': False, 'new_source_generation_after_worker_qualified': False,
        'production_default_activated': False, 'overall_deadline_seconds': overall_seconds,
        'operation_deadline_ceiling_seconds': 400,
        'paired_receiving_deadline_ceiling_seconds': 580, 'local_pool_bounded_by_host_envelope': True,
        'shares_host_pid_state': False, 'scheduler_configuration': scheduler.config.persisted_dict(),
        'container_resource_authority_pin': shared.pin(authority_raw)}
    index = registry = connection = selected = None
    base._write(output / 'container-resource-authority.json', authority)
    global_memory = _ParentRSSMonitor()

    def remaining(cap=400):
        seconds = deadline - time.monotonic()
        native.require(seconds > 0, 'full300 chunk qualification deadline exceeded')
        return min(cap, seconds)

    def options():
        return {'scheduler': scheduler, 'timeout_seconds': remaining(), 'memory_mb': 1024,
                'admission_timeout_seconds': min(30, remaining())}

    def paired_options():
        return {'scheduler': scheduler, 'timeout_seconds': remaining(580), 'memory_mb': 1024,
                'admission_timeout_seconds': min(30, remaining(580))}

    def phase(name, action):
        row = {'name': name, 'status': 'running'}
        report['phases'].append(row)
        base._progress(output / 'progress.json', report)
        begin = time.monotonic()
        try:
            value = action()
            row['status'] = 'completed'
            return value
        except BaseException as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic() - begin
            base._progress(output / 'progress.json', report)

    try:
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
        selected = native._pins(output, sorted(set(resume._implementation()['files'])
            | set(delta._implementation()['files']) | set(successor._implementation()['files']) | {
            MODULE, custody.__name__, deployment.__name__, shared.__name__, base.__name__,
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_chunks',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_chunk_scan',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_shared_chunks',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_shared_scan',
            'benchmarks.agent_supervisor.container_coding.run_codebase_full300_shared_docker',
            'ipfs_datasets_py.logic.backends.process',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_shared_successor_receiving',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_model',
            'ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_receiving',
            'benchmarks.agent_supervisor.container_coding.qualify_codebase_full300_shared',
            'benchmarks.agent_supervisor.container_coding.qualify_codebase_inventory_resume',
            'benchmarks.agent_supervisor.container_coding.run_codebase_inventory_resume_docker',
            'benchmarks.agent_supervisor.container_coding.audit_codebase_source_successor',
            'ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler',
            'benchmarks.agent_supervisor.container_coding.run_codebase_full300_shared_successor_docker'}))
        repository = output / 'repository'
        native._sources(repository)
        native.require(len([name for name in native._git(repository, 'ls-files', '-z').split('\0') if name]) == ENTRIES,
                       'original300 fixture population differs')
        index, registry, connection = phase('open_fresh_native_owners', lambda: native._open(output))
        preparation = {'repository_id': REPOSITORY_ID, 'limits': CodebaseScanLimits(512, 65536), 'exclusions': ('.runtime',)}
        previous = phase('publish_native_previous_source', lambda: index.prepare_current(repository,
            operation_id='initial', expected_head=None, **preparation, **options()))
        base._write(output / 'previous-publication-receipt.json', previous.to_dict())
        base._write(output / 'previous-source-manifest.json', index.load(previous.head.manifest_cid).to_dict())
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
                      (('calc.py', 'train'), ('tune.py', 'tune'), ('canary.py', 'canary'))]

        def fit(head, name, parent=None):
            before = 0 if parent is None else native._state(registry, parent)['completed_epochs']
            result = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=name, parent_version_id=parent,
                epochs=1, learning_rate=.002, seed=1729, **options())
            version = result.to_dict()['version_id']
            native.require(native._state(registry, version)['completed_epochs'] - before == 1, 'explicit setup epoch differs')
            report['new_fitting_epochs'] += 1
            _persist(output, name + '-training-record.json', result)
            return version

        parent = phase('explicit_root_training_one_epoch', lambda: fit(previous.head, 'root'))
        (repository / 'calc.py').write_bytes(b'def increment(n: int) -> int:\n    return n + 2\n')
        (repository / 'bulk000.py').unlink()
        (repository / 'bulk001.py').rename(repository / 'renamed.py')
        (repository / 'added.py').write_bytes(b'def added(n: int) -> int:\n    return n + 700\n')
        native._git(repository, 'add', '-A')
        native._git(repository, 'commit', '--quiet', '--no-verify', '-m', 'Explicit full300 native chunk successor')
        native.require(len([name for name in native._git(repository, 'ls-files', '-z').split('\0') if name]) == ENTRIES,
                       'full300 successor lost original population')
        current = phase('publish_native_current_source', lambda: index.prepare_current(repository,
            operation_id='successor', expected_head=previous.head, **preparation, **options()))
        base._write(output / 'current-publication-receipt.json', current.to_dict())
        manifest = index.load(current.head.manifest_cid)
        native.require(len(manifest.snapshot.entries) == ENTRIES and manifest.snapshot.max_entries == 512
            and manifest.snapshot.max_file_bytes == 65536 and current.head.generation == 2, 'original full300 capture profile differs')
        base._write(output / 'current-source-manifest.json', manifest.to_dict())
        transition = phase('build_native_source_delta', lambda: delta.build_current_codebase_source_delta(
            index, repository, previous_head=previous.head, expected_head=current.head, **options()))
        _persist(output, 'source-delta.json', transition)
        child = phase('explicit_child_training_one_epoch', lambda: fit(current.head, 'child', parent))
        versions, frozen = {'root': parent, 'child': child}, _states(registry, {'root': parent, 'child': child})
        frozen_heads = _variant_heads(registry, versions)
        report['variant_heads_before'] = frozen_heads
        report['registry_owner_observations'].append({'event':'after_explicit_setup', **_registry_owner(registry)})
        native.require(frozen['root']['completed_epochs'] == 1 and frozen['child']['completed_epochs'] == 2
            and all(row['latent_width'] == 8 and row['feature_columns'] == 53 for row in frozen.values()),
            'native CPU structural8D setup state or feature basis differs')
        base._write(output / 'checkpoint-states-before.json', frozen)
        completed, selected_successors = {}, {}
        with shared.no_fit(report), custody._no_install_guards(report):
            for mode in MODES:
                optimized = mode != 'reference'
                scan, records = _protocol(mode)
                observer = _RSSObserver(output, mode)
                with observer:
                    selection = None
                    if mode == 'legacy_default':
                        root = phase('start_fresh_' + mode + '_root', lambda: scan.start_current_codebase_chunked_scan(
                            index, repository, expected_head=current.head, registry=registry, version_id=child,
                            limits=resume.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=PAGE_ENTRIES),
                            optimized=optimized, **options()))
                    else:
                        selection, root = phase('select_' + mode + '_direct_successor',
                            lambda: _start_selected_shared_root(mode, index, repository,
                                source_delta=transition, registry=registry, previous_version_id=parent,
                                version_id=child, options=options()))
                        _persist(output, mode + '-successor-selection.json', selection)
                        selected_successors[mode] = selection
                    _persist(output, mode + '-root.json', root)
                    cursor, pages = None, []
                    for number in range(1, PARENT_PAGES + 1):
                        page = phase('infer_' + mode + '_page_' + str(number), lambda: scan.scan_current_codebase_page_chunked(
                            index, repository, root=root, registry=registry, cursor=cursor, **options()))
                        pages.append(page)
                        cursor = records.CodebaseScanChunkedCursor.from_page(root, page)
                        _export_page(output, mode, number, page, index)
                    native.require(cursor is not None and cursor.next_offset == 128, 'full300 fresh-process cursor missing')
                    native._close(registry, connection)
                    index = registry = connection = None
                    result_path, request_path = output / (mode + '-resume-result.json'), output / (mode + '-resume-request.json')
                    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
                    with scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=512,
                            child_process_slots=1, timeout=min(30, remaining()),
                            request_id='full300-shared-' + mode + '-fresh-process') as orchestration:
                        request = {'schema': RESUME_REQUEST, 'output': str(output), 'mode': mode, 'root_cid': root.artifact_cid,
                            'selection_cid': None if selection is None else selection.artifact_cid,
                            'source_delta_cid': transition.artifact_cid,
                            'cursor': cursor.to_dict(), 'versions': versions, 'checkpoint_states': frozen,
                            'expected_head': current.head.to_dict(), 'authority': str(host_configuration),
                            'authority_sha256': hashlib.sha256(authority_raw).hexdigest(), 'orchestration': orchestration.to_dict(),
                            'parent_identity': custody._process(os.getpid()), 'result': str(result_path),
                            'max_pages': RESUME_PAGES, 'timeout_seconds': remaining(1800)}
                        base._write(request_path, request)
                        digest = hashlib.sha256(shared.read_absolute(request_path, 1_048_576)).hexdigest()
                        command = [sys.executable, '-B', '-m', MODULE, '--resume-request', str(request_path),
                                   '--resume-request-sha256', digest]
                        with (output / (mode + '-resume.log')).open('xb') as log:
                            returncode, identity, cleanup = phase('fresh_process_' + mode + '_resume',
                                lambda: _launch_resume(command, log,
                                    min(remaining(1820), request['timeout_seconds'] + 20), orchestration))
                    native.require(returncode == 0, 'fresh chunk continuation failed')
                    resumed = json.loads(shared.read_absolute(result_path, 1_048_576))
                    native.require(resumed['selection_cid'] == (None if selection is None else selection.artifact_cid)
                        and resumed['source_delta_cid'] == transition.artifact_cid,
                        'fresh resume selection/delta input join differs')
                    native.require(resumed['qualified'] and resumed['request_sha256'] == digest
                        and resumed['post_setup_fit_attempt_count'] == resumed['new_fitting_epochs']
                        == resumed['receive_forward_attempt_count'] == 0 and resumed['checkpoint_states_after'] == frozen
                        and resumed['pid'] == identity['pid'] and resumed['process_identity']['birth'] == identity['birth'],
                        'fresh chunk continuation evidence differs')
                    report['fresh_resume_processes'].append({'mode': mode, 'pid': resumed['pid'],
                        'process_identity': identity, 'request_sha256': digest, 'result': result_path.name,
                        'launch_group_cleanup': cleanup})
                    shared.assert_clean(scheduler, owner_pids=[os.getpid(), resumed['pid']])
                    index, registry, connection = phase('cold_reopen_after_' + mode, lambda: native._open(output))
                    report['registry_owner_observations'].append({'event':'reopen_after_' + mode + '_resume',
                        **_registry_owner(registry)})
                    for number, descriptor in enumerate(resumed['pages'], PARENT_PAGES + 1):
                        page = scan.load_codebase_scan_chunked_page(index.artifacts, descriptor['page_cid'])
                        native.require(records.page_descriptor(page) == descriptor, 'fresh full300 page descriptor differs')
                        pages.append(page)
                        # Child persisted the ordinary page/chunk exports. Join
                        # each back to this process's independently read CAS.
                        _verify_page_exports(output, mode, number, page, index)
                    with _no_forward(report):
                        completion = phase('complete_all_' + mode + '_pages', lambda: scan.complete_current_codebase_chunked_scan(
                            index, repository, root=root, registry=registry, tail_page_cid=pages[-1].artifact_cid, **options()))
                        if mode == 'legacy_default':
                            phase('receive_current_' + mode + '_completion', lambda: scan.validate_current_codebase_chunked_scan_completion(
                                completion, index, repository, root=root, registry=registry, **options()))
                        else:
                            def late_read_only_callback(selected_record, completed_record):
                                native.require(selected_record is selection and completed_record is completion
                                    and index.current(REPOSITORY_ID) == current.head
                                    and _states(registry, versions) == frozen,
                                    'late callback exact source/weights/Adam observation differs')
                            warm = phase('paired_' + mode + '_warm_receive', lambda: _receive_selected_pair(
                                mode, 'warm', selection, completion, index, repository,
                                root=root, registry=registry, versions=versions, frozen_states=frozen,
                                state_reader=_states, options=paired_options(), late_callback=late_read_only_callback,
                                no_forward_guard=_no_forward, no_fit_guard=shared.no_fit,
                                rss_monitor_factory=_ParentRSSMonitor,
                                process_identity_reader=custody._process, report=report))
                            native.require(warm['variant_heads_before'] == warm['variant_heads_after'] == frozen_heads,
                                'warm paired receiving changed setup variant heads')
                            warm_path = output / (mode + '-warm-receive.json')
                            base._write(warm_path, warm)
                            report['paired_receiver_phase_files'].append({'mode':mode,'stage':'warm',
                                'file':warm_path.name,'pin':shared.pin(shared.read_absolute(warm_path,1_048_576))})
                    _persist(output, mode + '-completion.json', completion)
                    if mode != 'legacy_default':
                        native.require(_states(registry,versions) == frozen and _variant_heads(registry,versions) == frozen_heads,
                            'parent checkpoint/variant heads changed before cold receiving')
                        owner_before_close = _registry_owner(registry)
                        report['registry_owner_observations'].append({'event':'before_' + mode + '_cold_close',**owner_before_close})
                        phase('close_parent_before_' + mode + '_cold_receive', lambda: native._close(registry,connection))
                        index = registry = connection = None
                        cold_request_path = output / (mode + '-cold-receive-request.json')
                        cold_result_path = output / (mode + '-cold-receive-result.json')
                        cold_rss_path = output / (mode + '-cold-receive-rss.json')
                        with scheduler.acquire(lane=ResourceLane.ORCHESTRATION,cpu_slots=1,memory_mb=512,
                                child_process_slots=1,timeout=min(30,remaining()),
                                request_id='full300-paired-' + mode + '-cold-receive') as orchestration:
                            cold_request = {'schema':COLD_REQUEST,'output':str(output),'mode':mode,
                                'selection_cid':selection.artifact_cid,'source_delta_cid':transition.artifact_cid,
                                'root_cid':root.artifact_cid,'completion_cid':completion.artifact_cid,
                                'pages':completion.to_dict()['pages'],'versions':versions,'checkpoint_states':frozen,
                                'expected_head':current.head.to_dict(),'authority':str(host_configuration),
                                'authority_sha256':hashlib.sha256(authority_raw).hexdigest(),
                                'orchestration':orchestration.to_dict(),'parent_identity':custody._process(os.getpid()),
                                'result':str(cold_result_path),'rss_result':str(cold_rss_path),
                                'timeout_seconds':remaining(580),
                                'generation_inputs_sha256':hashlib.sha256(shared.read_absolute(output / 'generation-inputs.json',1_048_576)).hexdigest()}
                            native.require(set(cold_request) == COLD_REQUEST_FIELDS,
                                'parent cold request does not have the closed input contract')
                            base._write(cold_request_path,cold_request)
                            cold_request_pin = shared.pin(shared.read_absolute(cold_request_path,1_048_576))
                            command = [sys.executable,'-B','-m',MODULE,'--cold-receive',str(cold_request_path),
                                '--cold-receive-sha256',cold_request_pin['sha256']]
                            cold_guard = _ColdRSSObserver(output,mode)
                            with cold_guard, (output / (mode + '-cold-receive.log')).open('xb') as log:
                                cold_returncode,cold_identity,cold_cleanup = phase('fresh_process_' + mode + '_cold_receive',
                                    lambda: _launch_cold_receiver(command,log,
                                        min(remaining(600),cold_request['timeout_seconds'] + 20),orchestration))
                                cold_guard.bind(cold_cleanup)
                            cold_guard_memory = cold_guard.document()
                            guard_path = output / (mode + '-cold-launch-rss.json')
                            base._write(guard_path,cold_guard_memory)
                            launch_record = {'mode':mode,'pid':cold_identity['pid'],'process_identity':cold_identity,
                                'request_sha256':cold_request_pin['sha256'],'request_pin':cold_request_pin,
                                'result':cold_result_path.name,'rss_result':cold_rss_path.name,
                                'launch_group_cleanup':cold_cleanup,'launch_rss_file':guard_path.name,
                                'launch_rss_pin':shared.pin(shared.read_absolute(guard_path,1_048_576)),
                                'parent_native_owners_closed_before_spawn':True,'parent_reopened_after_actual_drain':False,
                                'parent_owner_before_close':owner_before_close}
                            report['fresh_cold_receiver_processes'].append(launch_record)
                            native.require(cold_returncode == 0,'actual fresh cold receiving failed')
                            received,received_pins = _join_cold_receive(output,mode,cold_request,cold_request_pin,
                                cold_identity,cold_cleanup,cold_guard_memory,selection,root,completion,versions,frozen,frozen_heads)
                            launch_record['raw_output_pins'] = received_pins
                            native.require(shared.pin(shared.read_absolute(cold_request_path,1_048_576)) == cold_request_pin,
                                'actual cold request changed during launch/join')
                        shared.assert_clean(scheduler,owner_pids=[os.getpid(),cold_identity['pid']])
                        index,registry,connection = phase('reopen_after_' + mode + '_cold_receive',lambda: native._open(output))
                        reopened_owner = _registry_owner(registry)
                        native.require(received['registry_owner_before']['owner_generation'] == owner_before_close['owner_generation'] + 1
                            and reopened_owner['owner_generation'] == received['registry_owner_after']['owner_generation'] + 1
                            and _states(registry,versions) == frozen and _variant_heads(registry,versions) == frozen_heads
                            and index.current(REPOSITORY_ID) == current.head,
                            'actual parent/cold/reopen generations or frozen source/checkpoint/heads differ')
                        launch_record.update(parent_reopened_after_actual_drain=True,parent_owner_after_reopen=reopened_owner)
                        report['registry_owner_observations'].append({'event':'reopen_after_' + mode + '_cold_receive',**reopened_owner})
                        report['paired_receiver_phase_records'].append(received['paired_receiver_phase_record'])
                        report['paired_receiver_phase_files'].append({'mode':mode,'stage':'cold',
                            'file':mode + '-cold-receive-phase.json','pin':received_pins['phase']})
                    coverage = completion.to_dict()['coverage']
                    native.require(coverage['inventory_entries'] == ENTRIES and coverage['pages'] == len(pages) == PAGES
                        and coverage['inferred_rows'] == 294 and coverage['projection_records'] == 588
                        and coverage['dispositions'] == EXPECTED_DISPOSITIONS, 'full300 numerical completion coverage differs')
                    report['new_' + mode + '_scan_pages'] = len(pages)
                    report[mode + '_coverage'] = coverage
                    report['actual_numerical_requests_by_mode'][mode] = coverage['chunks']
                    completed[mode] = (root, completion, pages)
                parent_memory = observer.document()
                base._write(output / (mode + '-parent-rss.json'), parent_memory)
                resume_memory = json.loads(shared.read_absolute(output / (mode + '-resume-rss.json'), 1_048_576))
                report['rss_observation_files'][mode] = {'parent': mode + '-parent-rss.json',
                    'resume': mode + '-resume-rss.json'}
                report['actual_rss_by_mode'][mode] = _join_rss(index, pages, mode,
                    [parent_memory, resume_memory], output, root)
                report['actual_mode_measurements'][mode] = _measurements(index, pages, mode)
            report['all_page_opt_out_equivalence'] = phase('compare_shared_default_reference_equal294',
                lambda: _compare_pages(index, completed['default'][2], completed['reference'][2],
                    'default', 'reference', exact=True))
            report['legacy_default_equivalence'] = phase('compare_legacy_shared_default_equal294',
                lambda: _compare_pages(index, completed['legacy_default'][2], completed['default'][2],
                    'legacy_default', 'default', exact=False))
            _persist(output, 'scan-root.json', completed['default'][0])
            _persist(output, 'scan-completion.json', completed['default'][1])
            after = _states(registry, versions)
            report['variant_heads_after'] = _variant_heads(registry, versions)
            native.require(after == frozen and report['variant_heads_after'] == frozen_heads
                and index.current(REPOSITORY_ID) == current.head, 'chunks/receiving changed model, Adam, variant heads or source')
            phase('refuse_precancelled_and_wrong_root_full300_pairs',lambda: _receiving_refusal_controls(
                selected_successors['default'], completed['default'][0], completed['default'][1],
                index,repository,registry,versions,frozen,options,report))
            base._write(output / 'checkpoint-states-after.json', after)
            _require_receiving_exports(output,report)
            native._require_pins(output, selected)
            absent = {'intent_database_absent': not (output / 'private' / 'intent.duckdb').exists(),
                'worker_launch_directory_absent': not (output / 'private' / 'launch').exists(),
                'worker_worktrees_absent': not (repository / '.git' / 'worktrees').exists()}
            native.require(all(absent.values()) and all(value == 0 for value in report['no_install_guards']['counters'].values())
                and report['new_fitting_epochs'] == 2 and report['post_setup_fit_attempt_count']
                == report['receive_forward_attempt_count'] == 0
                and len({row['pid'] for row in report['fresh_resume_processes']}) == 3,
                'chunk setup, receiving, fresh process or no-install accounting differs')
            report['no_install_guards'].update(absent)
            base._write(output / 'no-install-guards.json', report['no_install_guards'])
            phases = report['paired_receiver_phase_records']
            native.require(len(phases) == 4 and {(row['mode'],row['stage']) for row in phases}
                == {('default','warm'),('default','cold'),('reference','warm'),('reference','cold')}
                and all(row['accepted'] is True and row['native_scope_exit_success'] is True
                    and row['callback_invocations'] == row['successful_close_calls'] == 1 for row in phases)
                and len(report['fresh_cold_receiver_processes']) == 2
                and len({(row['process_identity']['pid'],row['process_identity']['birth']) for row in
                    report['fresh_resume_processes'] + report['fresh_cold_receiver_processes']}) == 5
                and all(row['parent_native_owners_closed_before_spawn'] is True
                    and row['parent_reopened_after_actual_drain'] is True for row in report['fresh_cold_receiver_processes'])
                and report['receiving_extra_numerical_attempt_count'] == report['receiving_mutation_attempt_count'] == 0
                and len(report['controls']) == 2 and all(row['observed'] == 'refuse' for row in report['controls']),
                'complete fresh paired receiving/parent-close/cold-drain/control population differs')
            report.update(qualified=True, full_campaign_qualified=False,
                warm_paired_receiver_qualified=True,cold_paired_receiver_qualified=True,
                paired_shared_successor_receiving_qualified=True,full300_refusal_controls_qualified=True,
                complete_scan_qualified=True, large300_member_scan_qualified=True,
                actual294_numerical_rows_per_mode_qualified=True, current_shared_and_legacy_receiving_qualified=True,
                actual_sampled_rss_join_qualified=True, shared_transport_streaming_qualified=True,
                selected_producers_unchanged=True, native_source_generation_advanced_by_worker=False)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary = sys.exc_info()[0] is not None
        memory_error = None
        try:
            report['parent_rss_observation'] = global_memory.finish()
        except BaseException as error:
            report.update(qualified=False, memory_observation_error_type=type(error).__name__,
                memory_observation_error=str(error))
            memory_error = error
        _finalize_campaign(output, report, registry=registry, connection=connection,
            scheduler=scheduler, selected_producers=selected, started=started,
            primary_error_active=primary or memory_error is not None)
        if memory_error is not None and not primary:
            raise memory_error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--overall-seconds', type=int, default=5950)
    parser.add_argument('--host-configuration', default='/results/container-resource-authority.json')
    parser.add_argument('--resume-request')
    parser.add_argument('--resume-request-sha256')
    parser.add_argument('--cold-receive')
    parser.add_argument('--cold-receive-sha256')
    args = parser.parse_args()
    if args.cold_receive:
        native.require(args.cold_receive_sha256 is not None and args.resume_request is None
            and args.resume_request_sha256 is None and args.output is None, 'exact standalone cold receiving request required')
        _cold_receive(args.cold_receive,args.cold_receive_sha256)
    elif args.resume_request:
        native.require(args.resume_request_sha256 is not None and args.output is None
            and args.cold_receive_sha256 is None, 'exact standalone chunk request required')
        _resume(args.resume_request, args.resume_request_sha256)
    else:
        native.require(args.output is not None and args.resume_request_sha256 is None
            and args.cold_receive_sha256 is None, 'fresh full300 selected successor output required')
        run(args.output, overall_seconds=args.overall_seconds, host_configuration=args.host_configuration)


if __name__ == '__main__':
    main()
