"""Real checkpoint/read-only validation under authored native pressure telemetry.

This control does not induce host pressure or claim observed host PSI recovery.
It uses an isolated real scheduler, elapsed wall time, actual checkpoint/GTE and
captured-header preparation/Z3; the warm calls never execute neural inference.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from test.api.test_header_intent_applicability import case  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_context import selected_config  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_source384_warm_recovery as recovery
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import EXTENDED_SOURCE384_PROFILE
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


def test_real_native_admission_recovers_after_ninety_seconds_without_inference_replay(
        case, selected_config, monkeypatch, record_property):
    import pytest
    from ipfs_datasets_py.logic.software_contracts import codebase_resources, codebase_source_units_384
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
    c = case
    config = json.loads(selected_config.read_bytes())
    config.update(schema='terminal-source384-config@2', header_applicability=c.config['header_applicability'])
    selected_config.write_bytes(source._raw(config))
    hashes = {name: row['sha256'] for name, row in c.manifest['payload']['sources'].items()}
    prepared_at = time.monotonic()
    receipt = source.prepare_source384_context(repository=c.repo, source_hashes=hashes,
        output=c.root/'source384', config_path=selected_config, timeout_seconds=180.,
        scheduler=c.scheduler, intent_binding={'contract': c.contract, 'manifest_cid': local.content_identity(c.manifest)})
    preparation_seconds = time.monotonic() - prepared_at
    inference_path = Path(receipt['output'])/'inference.json'
    inference_before = inference_path.read_bytes()
    inference = json.loads(inference_before)
    assert inference['native_worker_executed'] is True and inference['inference_executed'] is True
    assert inference['report']['output']['model_loads'] == 1
    monkeypatch.setattr(codebase_source_units_384, '_worker', lambda *a, **k: pytest.fail('warm neural inference'))
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    healthy = ProofHostResources(8, 16384, 16384)
    pressure_until = [None]
    def sample():
        return replace(healthy, memory_stall_percent=11.) if (pressure_until[0] is not None
            and time.monotonic() < pressure_until[0]) else healthy
    config = resources.ResourceSchedulerConfig.for_proof_host(state_path=c.root/'warm-scheduler.json',
        proof_resource_sampler=sample, proof_resource_profile='local-benchmark@1',
        proof_memory_stall_percent=10., proof_recovery_enabled=True, proof_recovery_grants=1,
        lane_reservations={}, auto_renew_leases=False, poll_interval_seconds=.02)
    scheduler = resources.GlobalResourceScheduler(config)
    monkeypatch.setattr(codebase_resources, 'get_global_resource_scheduler', lambda: scheduler)
    began = time.monotonic()
    pressure_until[0] = began + 91.
    result = recovery.observe_warm_context(repository=c.repo, expected_receipt=receipt,
        profile=EXTENDED_SOURCE384_PROFILE, policy=recovery.POLICY, deadline_monotonic=began+180.)
    elapsed = time.monotonic()-began
    recovery.validate_observation(result, profile=EXTENDED_SOURCE384_PROFILE, policy=recovery.POLICY,
        producer_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest())
    assert 95. <= elapsed < 180.
    assert len(result['attempts']) == 2 and result['status'] == 'validated'
    first, second = result['attempts']
    assert 89. <= first['elapsed_seconds'] <= 95.
    assert first['status'] == 'memory_admission_timeout' and second['status'] == 'validated'
    assert first['admission']['last_sample']['host']['memory_stall_percent'] == 11.
    assert first['admission']['last_sample']['thresholds']['memory_stall_percent'] == 10.
    assert second['timeout_seconds'] <= 85.1
    assert inference_path.read_bytes() == inference_before
    assert json.loads((Path(receipt['output'])/'receipt.json').read_bytes()) == receipt
    assert all(hashlib.sha256((c.repo/name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    snapshot = scheduler.snapshot()
    assert snapshot['active_lease_count'] == snapshot['waiting_request_count'] == 0
    record_property('authored_pressure_real_elapsed', json.dumps(dict(
        telemetry='authored_memory_stall_11_percent_then_zero', actual_host_pressure_induced=False,
        initial_preparation_seconds=preparation_seconds, warm_observation=result,
        warm_model_loads=0, initial_model_loads=1, leases_after=0, waiting_after=0), sort_keys=True))
