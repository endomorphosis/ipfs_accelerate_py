"""Finite kernel-peer packet delivery with content-free, offline accounting.

Root runs this inside its pinned offline container. Native fixture transport is
authored. This module never invokes a provider, target code, or a validation
command. The comparison baseline is not an alternative live delivery route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import threading

from .terminal_world_packet_replay_experiment import CLEANUP_SCOPE, MAX_FIXTURE_BYTES, _world_private_fixture
from .terminal_world_prelaunch_experiment import _bounded, _wire
from .terminal_world_worker_handoff_experiment import _line, _refused

SCHEMA = "terminal-world-worker-packet-offline-experiment@1"
SCOPE = "authenticated_worker_packet_delivery_and_authored_local_comparison_only"
MAX_REPORT_BYTES = 256 * 1024
MAX_TEXT_BYTES = 512 * 1024
MAX_COUNTER = (1 << 63) - 1
TOKENIZER_RANKS_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
CASE_COUNTS = (1, 4, 8)
WORKER_ENTRY = Path("/usr/local/libexec/ipfs-world-packet-delivery-worker")
WORKER_ARGV = ("/usr/bin/sudo", "-n", "-u", "benchmarkworker", "-g", "supervisor", "--", str(WORKER_ENTRY))
WORKER_PEER_ARGV = ("/opt/ipfs-supervisor/venv/bin/python", "-I", str(WORKER_ENTRY))
WORKER_ENTRY_SOURCE = '''#!/opt/ipfs-supervisor/venv/bin/python -I
import ctypes,os,sys
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0:raise OSError("no_new_privileges_unavailable")
os.environ.update({"IPFS_ACCEL_SKIP_CORE":"1","IPFS_KIT_DISABLE":"1","PYTHONDONTWRITEBYTECODE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","RAYON_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false"})
sys.path[:0]=["/opt/ipfs-supervisor/source","/opt/ipfs-supervisor/datasets","/opt/ipfs-supervisor/kit"]
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_packet_delivery import worker_packet_client_main
raise SystemExit(worker_packet_client_main())
'''
FALSE = dict(provider_calls=0, model_calls=0, target_code_executed=False,
    training_executed=False, validation_commands_executed=False, actual_quack_tcp_verified=False,
    provider_ingestion_verified=False, provider_usage_verified=False, billing_verified=False,
    token_savings_established=False, total_token_savings_measured=False,
    proof_authority=False, execution_authority=False, completion_authority=False,
    task_completed=False, world_dispatch_ready=False, provider_launch=False,
    compiler_signature_verified=False, semantic_correctness_verified=False,
    whole_repository_observed=False, worker_workspace_verified=False,
    full_container_containment_verified=False, container_image_independently_verified=False,
    database_access_granted=False, raw_content_exported=False, live_terminal_bench_run=False,
    actual_model_route_fallback=False)
CHECKS = frozenset({"trusted_selected_source_provisioning", "original_native_projection_current",
    "compile_after_allocated_root_transition", "old_gate_and_allocation_stay_stale",
    "complete_native_business_projection_preserved", "all_required_reference_bytes_preserved",
    "same_source_and_required_facts_in_both_arms", "zero_native_validation_commands_preserved",
    "original_compiled_binding_current", "real_literal_popen_registered",
    "actual_worker_uid_gid_birth_verified", "exact_packet_received_and_acknowledged",
    "no_worker_broker_database_or_completion_grant", "stale_native_body_aborts_while_worker_alive",
    "owned_worker_reaped", "complete_get_ack_refusal_population", "native_contexts_closed"})
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_world_worker_packet_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_worker_handoff_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_packet_replay_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_prelaunch_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_source_view_experiment.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_worker_packet.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_worker_packet_delivery.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_worker_observer.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_worker_handoff.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_prelaunch.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_task_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_worktree_allocation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_session.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_index.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_context_compiler.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet_replay.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/context/context_contracts.py",
    "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
    "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py",
    "test/api/test_current_source_task_admission.py",
    "test/api/test_current_source_worktree_allocation.py",
    "test/api/test_repository_world_prelaunch.py",
)
_LOCK = threading.Lock()
GUIDANCE = "Authored comparison transcript only. Discover the pinned descriptor, request get_packet, verify the returned digest, and acknowledge the exact bytes. No provider receives this transcript.\n"


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def report_sha256(report):
    """FULL canonical report digest, including the internal seal."""
    return _sha(_wire(report))


def _pins():
    root = Path(__file__).resolve().parents[3]
    files = {}
    for name in SOURCE_PATHS:
        path = root / name
        _need(path.resolve(strict=True) == path, "source_pin_alias")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_size <= 2 * 1024**2, "source_pin_bound")
            raw = bytearray()
            while len(raw) < before.st_size:
                part = os.read(fd, min(65536, before.st_size - len(raw)))
                _need(bool(part), "source_pin_short_read")
                raw.extend(part)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
            _need(not os.read(fd, 1) and identity(before) == identity(os.fstat(fd)) == identity(path.lstat())
                and path.resolve(strict=True) == path, "source_pin_changed")
            files[name] = dict(bytes=len(raw), sha256=_sha(raw))
        finally:
            os.close(fd)
    return dict(scope="listed_source_files_only_not_transitive_provenance", files=files)


def _profile():
    _need(os.getuid() == os.geteuid() == 1000 and os.getgid() == os.getegid() == 1000,
        "actual_owner_identity_required")
    worker = pwd.getpwnam("benchmarkworker")
    _need(worker.pw_uid == 1001 and worker.pw_gid == 1000, "preinstalled_worker_identity_required")
    for path in (Path(WORKER_ARGV[0]), WORKER_ENTRY):
        _need(path.resolve(strict=True) == path, "trusted_worker_launcher_alias")
        for parent in (path, *path.parents):
            info = parent.lstat()
            _need(info.st_uid == 0 and not info.st_mode & 0o022, "trusted_worker_launcher_not_root_controlled")
        info = path.lstat()
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_mode & 0o111,
            "trusted_worker_launcher_not_regular")
    _need(WORKER_ENTRY.read_bytes() == WORKER_ENTRY_SOURCE.encode(), "trusted_worker_launcher_source_differs")
    return dict(owner_uid=1000, owner_gid=1000, worker_uid=1001, worker_gid=1000,
        launcher_sha256=_sha(WORKER_ENTRY_SOURCE.encode()),
        literal_popen_argv_sha256=_sha(_wire(list(WORKER_ARGV))),
        worker_peer_argv_sha256=_sha(_wire(list(WORKER_PEER_ARGV))),
        namespaces={name: os.readlink("/proc/self/ns/" + name) for name in ("pid", "mnt", "net")})


def _tokenizer(cache_file):
    from .terminal_source_view_experiment import local_token_counter
    counter, metadata = local_token_counter(cache_file)
    result = dict(available=counter is not None, encoding="o200k_base" if counter is not None else None,
        ranks_sha256=metadata.get("ranks_sha256"), reason=metadata.get("reason"),
        count_scope="exact_local_encoding_of_authored_text_only", provider_compatibility_verified=False,
        provider_usage=False, message_framing_included=False, download_performed=False)
    return counter, result


def _measurement(text, counter):
    _need(type(text) is str and len(text.encode()) <= MAX_TEXT_BYTES, "measurement_text_bound")
    raw = text.encode("utf-8", "strict")
    tokens = counter(text) if counter is not None else None
    _need(tokens is None or type(tokens) is int and 0 <= tokens <= MAX_COUNTER, "local_token_count_invalid")
    return dict(bytes=len(raw), sha256=_sha(raw), local_o200k_tokens=tokens)


def _comparison(baseline, candidate, common_trace, counter):
    measured = dict(baseline_initial=_measurement(baseline, counter),
        candidate_initial=_measurement(candidate, counter),
        common_authored_protocol_text=_measurement(common_trace, counter),
        baseline_authored_trace=_measurement(baseline + common_trace, counter),
        candidate_authored_trace=_measurement(candidate + common_trace, counter))
    old, new = measured["baseline_initial"], measured["candidate_initial"]
    disposition = ("token_count_unavailable" if old["local_o200k_tokens"] is None else
        "compiled_smaller" if new["bytes"] < old["bytes"] and new["local_o200k_tokens"] < old["local_o200k_tokens"]
        else "raw_comparison_preferred_not_smaller")
    return dict(measurements=measured, offline_comparison_disposition=disposition,
        service_delivery_mode="compiled_packet_only", actual_model_route_fallback=False,
        trace_scope="one_authored_initial_prompt_plus_same_discovery_guidance_and_helper_receipts_once",
        source_bytes_already_in_initial_prompt=True, ipc_frames_added_to_model_tokens=False,
        protocol_displayed_to_model=None, complete_native_total_tokens=None, billed_tokens=None)


def _source(count):
    _need(type(count) is int and count in CASE_COUNTS, "unsupported_source_case")
    return ("# Authored inert fixture; never imported or executed.\n" + "\n".join(
        f"def helper_{index}(value):\n    return value + {index + 1}\n" for index in range(count))).encode()


def _fetch(process):
    process.stdin.write(b"get_packet\n"); process.stdin.flush()
    raw = _line(process.stdout, timeout=15)
    value = json.loads(raw)
    _need(type(value) is dict and _wire(value) + b"\n" == raw, "worker_receipt_not_canonical")
    return value, raw


def _cycle(fixture, count, counter):
    import pytest
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler, EvidenceValuePolicy
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget, ContextReference, ContextTier
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_prelaunch import capture_repository_world_prelaunch_assignment
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_handoff import capture_repository_world_worker_handoff
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_observer import RepositoryWorldWorkerObserver
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_packet import capture_repository_world_task_projection, capture_repository_world_compiled_packet_binding
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_packet_delivery import RepositoryWorldWorkerPacketDelivery, FALSE_FIELDS, RECEIPT_SCHEMA
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import prepare_repository_world_packet, READONLY_GUIDANCE
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import canonical_json_bytes as context_json_bytes
    from test.api.test_current_source_worktree_allocation import _allocation_case
    from test.api.test_repository_world_prelaunch import provision_authored_prelaunch_sources

    checks = {}
    process = service = observer = handoff = None
    source = _source(count)
    with pytest.MonkeyPatch.context() as patch:
        with _allocation_case(fixture, patch, defer_source_binding=True, source_files={b"module.py": source}) as case:
            native = case.native
            _need(type(native.server.transport) is FakeQuackTransport, "authored_native_transport_profile_differs")
            native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
            provision = provision_authored_prelaunch_sources(case)
            checks["trusted_selected_source_provisioning"] = provision == dict(selected_files=2,
                bytes_unchanged=True, executable_bits_unchanged=True)
            projection = capture_repository_world_task_projection(task_admission=native.handle)
            projection_metadata = projection.require_current()
            task_text = projection.task_text
            public = projection.public_task()
            checks["original_native_projection_current"] = projection_metadata["task_text_sha256"] == _sha(task_text.encode())
            assignment = capture_repository_world_prelaunch_assignment(task_admission=native.handle,
                worktree_admission=case.guard, prompt=task_text)
            try:
                handoff = capture_repository_world_worker_handoff(assignment=assignment,
                    worker_uid=1001, worker_gid=1000, timeout_seconds=150)
                handoff.apply_readonly_transition()
                checks["old_gate_and_allocation_stay_stale"] = _refused(assignment.require_current) and _refused(case.guard.require_current)
                with RepositorySourceSession(native.workspace, fixture / "packet-source-state",
                        max_files=8, max_file_bytes=8192, max_total_bytes=65536, max_generations=1) as session:
                    checks["compile_after_allocated_root_transition"] = stat.S_IMODE(native.workspace.stat().st_mode) == 0o750
                    current_source = session.read(generation=1, path="module.py", representation="original")
                    _need(current_source["content"].encode() == source, "selected_source_bytes_differ")
                    core = projection.compile_core()
                    _need(core["allowed_effects"] == ("repository_read",), "readonly_effect_profile_differs")
                    source_ref = ContextReference(reference_id="authored-selected-source:module.py", kind="exact-selected-source",
                        tier=ContextTier.INVARIANT, referenced_content_id="sha256:" + _sha(source),
                        path="module.py", summary=source.decode(), metadata=dict(required=True,
                            coverage_ids=("authored-selected-source:coverage",), source_sha256=_sha(source),
                            source_binding_scope="original_native_selected_file_bytes_only"))
                    ir_refs = tuple(ContextReference(reference_id="authored:" + name, kind="authored-" + name,
                        tier=ContextTier.INVARIANT, referenced_content_id="sha256:" + _sha(text.encode()), summary=text,
                        metadata=dict(required=True, coverage_ids=(name + ":authored-coverage",), learned_inference=False))
                        for name, text in (("intent_ir", "Preserve the complete declared native task and literal source population."),
                            ("security_ir", "Read only; no target execution, SQL access, credential disclosure, or provider invocation."),
                            ("ui_ux_ir", "Preserve literal public signatures and uncertainty; do not claim user-facing correctness.")))
                    required = (projection.context_reference(), source_ref, *ir_refs)
                    compiler = ContextCompiler(ContextBudget(max_input_tokens=65536, reserved_output_tokens=0,
                        reserved_tool_tokens=0, max_items=64, max_item_bytes=32768, max_text_bytes=32768),
                        tokenizer=counter if counter is not None else lambda text: max(1, (len(text.encode()) + 3) // 4),
                        value_policy=EvidenceValuePolicy())
                    adapter = session.context_adapter(compiler)
                    compiled = adapter.compile(generation=1, query=" ".join(f"helper_{index}" for index in range(count)),
                        unresolved_question="Which exact authored inert definitions are in the selected file?",
                        required_evidence=required, required_reference_ids=tuple(sorted(ref.reference_id for ref in required)),
                        required_coverage_ids=tuple(sorted(value for ref in required for value in ref.coverage_ids)),
                        objective_id="authored-packet-delivery", objective_revision="sha256:" + _sha(source),
                        policy_id="readonly-worker-packet", policy_revision="read_only_observation@1",
                        goal=core["goal"], authority=core["authority"], scope=core["scope"], acceptance=core["acceptance"],
                        limit=8, max_bytes=32768)
                    selected = (*compiled.capsule.evidence, *compiled.capsule.expansion_references)
                    visible = {ref.reference_id: ref for ref in selected}
                    checks["all_required_reference_bytes_preserved"] = all(visible[ref.reference_id].canonical_bytes() == ref.canonical_bytes() for ref in required)
                    checks["complete_native_business_projection_preserved"] = json.loads(visible[required[0].reference_id].summary) == public
                    checks["zero_native_validation_commands_preserved"] = (
                        projection_metadata["relation_counts"]["validations"] == 0 and public["relations"]["validations"] == []
                        and projection_metadata["native_validation_coverage"] == "no_native_validation_commands_declared")
                    packet = prepare_repository_world_packet(adapter=adapter, compilation=compiled,
                        generation=1, task_id=native.task_cid, task_text=task_text,
                        unresolved_obligations=("No native validation commands are declared; no source semantic or provider completion has been verified.",),
                        validation_commands=(), validation_profile="read_only_observation@1", proof_obligations=(),
                        allowed_effects=core["allowed_effects"], write_paths=core["write_paths"], max_bytes=262144)
                    binding = capture_repository_world_compiled_packet_binding(projection=projection, handoff=handoff, packet=packet)
                    binding_metadata, packet_text = binding.delivery_material(handoff=handoff, original_prompt=task_text)
                    checks["original_compiled_binding_current"] = packet_text == packet.prompt
                    capsule_record = json.loads(context_json_bytes(compiled.capsule.provider_input_payload))
                    header = {key: value for key, value in capsule_record.items() if key not in {"evidence", "expansion_references"}}
                    baseline = READONLY_GUIDANCE + "\n" + _wire(dict(schema="authored-raw-inline-comparison@1", native_task_projection=public,
                        compile_core=core, compiled_context_header=header, evidence=[ref.to_record() for ref in selected],
                        world_availability=dict(status="unavailable", unknown_dimensions="not_projected", proof_authority=False),
                        task_summary="Observe the exact current task evidence; no execution is authorized.",
                        unresolved_obligations=["No native validation commands are declared; no source semantic or provider completion has been verified."],
                        validation_profile="read_only_observation@1", validation_commands=[], proof_obligations=[])).decode()
                    checks["same_source_and_required_facts_in_both_arms"] = all(
                        ref.summary.encode() == visible[ref.reference_id].summary.encode() for ref in required)
                    baseline_rendered, baseline_advisory = render_model_prompt(prompt=baseline, purpose="coding", workspace=native.workspace, world_packet=True)
                    candidate_rendered, candidate_advisory = render_model_prompt(prompt=packet_text, purpose="coding", workspace=native.workspace, world_packet=True)
                    _need(baseline_advisory == candidate_advisory, "comparison_advisory_differs")
                    observer = RepositoryWorldWorkerObserver(handoff=handoff, prompt=task_text,
                        expected_launcher_argv=WORKER_ARGV, expected_worker_argv=WORKER_PEER_ARGV, timeout_seconds=90)
                    service = RepositoryWorldWorkerPacketDelivery(observer=observer, binding=binding, timeout_seconds=90)
                    process = subprocess.Popen(list(WORKER_ARGV), cwd=native.workspace, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                    service.note_spawned(process)
                    checks["real_literal_popen_registered"] = process.poll() is None
                    initial = _wire(dict(descriptor=service.descriptor, expected_descriptor_sha256=service.descriptor_sha256)) + b"\n"
                    _need(len(initial) <= 8193, "worker_initial_frame_bound")
                    process.stdin.write(initial); process.stdin.flush()
                    positive, positive_raw = _fetch(process)
                    _need(positive.get("status") == "received_and_acknowledged", "actual_worker_packet_delivery_refused")
                    peer = positive["peer"]
                    checks["actual_worker_uid_gid_birth_verified"] = (peer["uid"] == 1001 and peer["gid"] == 1000
                        and peer["pid"] > 0 and peer["start_time_ticks"] > 0 and peer["root_pid"] == process.pid
                        and peer["root_start_time_ticks"] > 0 and positive["worker_peer_authenticated"] is True)
                    checks["exact_packet_received_and_acknowledged"] = (positive["prompt_sha256"] == _sha(packet_text.encode())
                        and positive["prompt_bytes"] == len(packet_text.encode()) and positive["packet_byte_identity_verified"] is True
                        and positive["acknowledgement_recorded"] is True)
                    checks["no_worker_broker_database_or_completion_grant"] = all(positive[name] is False for name in FALSE_FIELDS)
                    original_body = canonical_json_bytes(dict(native.native_task.body)).decode()
                    changed = dict(native.native_task.body); changed["title"] = "Authored stale packet delivery"
                    with native.server._lock:
                        native.server._connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=?",
                            [canonical_json_bytes(changed).decode(), native.task_cid])
                    try:
                        alive = process.poll() is None
                        negative, negative_raw = _fetch(process)
                        checks["stale_native_body_aborts_while_worker_alive"] = (alive and negative.get("schema") == RECEIPT_SCHEMA
                            and negative.get("status") == "refused" and negative.get("reason_code") == "delivery_refused"
                            and negative.get("worker_peer_authenticated") is False and negative.get("acknowledgement_recorded") is False
                            and _refused(lambda: binding.delivery_material(handoff=handoff, original_prompt=task_text)))
                    finally:
                        with native.server._lock:
                            native.server._connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=?", [original_body, native.task_cid])
                    process.stdin.close(); code = process.wait(timeout=5)
                    checks["owned_worker_reaped"] = code == 0
                    stopped = service.close(); service = None
                    observer.close(); observer = None
                    checks["complete_get_ack_refusal_population"] = (stopped["service_stopped"] is True
                        and stopped["accepted_connections"] == stopped["sent_responses"] == 3
                        and stopped["refused_responses"] == 1 and stopped["sent_packet_responses"] == 1
                        and stopped["acknowledged_packets"] == 1 and stopped["send_failures"] == 0
                        and stopped["sent_packet_bytes"] == len(packet_text.encode())
                        and stopped["sent_response_wire_bytes"] == stopped["sent_response_bytes"] + 12)
                    common = GUIDANCE + initial.decode() + "get_packet\n" + positive_raw.decode() + "get_packet\n" + negative_raw.decode()
                    comparison = _comparison(baseline_rendered, candidate_rendered, common, counter)
                    measurement = dict(source_sha256=_sha(source), selected_source_files=1,
                        selected_source_bytes=len(source), authored_function_count=count,
                        required_reference_count=len(required), structural_reference_count=sum(ref.reference_id.startswith("repository-symbol:") for ref in selected),
                        native_validation_count=0, native_validation_coverage="no_native_validation_commands_declared",
                        task_projection_sha256=_sha(task_text.encode()), binding_metadata_sha256=_sha(_wire(binding_metadata)),
                        packet_text=_measurement(packet_text, counter), required_source_text=_measurement(source.decode(), counter),
                        positive_receipt_sha256=_sha(positive_raw), negative_receipt_sha256=_sha(negative_raw), worker_peer=dict(peer),
                        ipc=dict(accepted_connections=stopped["accepted_connections"], sent_responses=stopped["sent_responses"],
                            refused_responses=stopped["refused_responses"], sent_packet_responses=stopped["sent_packet_responses"],
                            acknowledged_packets=stopped["acknowledged_packets"], sent_packet_bytes=stopped["sent_packet_bytes"],
                            sent_response_bytes=stopped["sent_response_bytes"], sent_response_wire_bytes=stopped["sent_response_wire_bytes"],
                            helper_stdin_bytes=len(initial) + 2 * len(b"get_packet\n"),
                            helper_stdout_bytes=len(positive_raw) + len(negative_raw), packet_body_counted_once=True,
                            byte_scope="actual_service_canonical_response_frames_and_exact_helper_transcript"),
                        comparison=comparison)
                handoff.close(); handoff = None
            finally:
                if process is not None and process.poll() is None:
                    process.terminate(); process.wait(timeout=5)
                if process is not None:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None and not stream.closed: stream.close()
                if service is not None: service.close()
                if observer is not None: observer.close()
                if handoff is not None: handoff.close()
    checks["native_contexts_closed"] = True
    _need(set(checks) == CHECKS and all(value is True for value in checks.values()), "worker_packet_check_failed")
    return dict(checks=checks, measurement=measurement)


def _failure(error):
    text = str(error)
    return dict(stage="trusted_worker_packet_fixture", error_type=type(error).__name__,
        reason_code=text if re.fullmatch(r"[a-z][a-z0-9_]{0,95}", text) else "unspecified_failure")


def run_experiment(*, scratch_root: Path, tokenizer_cache: Path | None = None):
    _need(_LOCK.acquire(blocking=False), "worker_packet_experiment_already_running")
    previous = Path.cwd()
    try:
        before = _pins()
        cases = []
        profile = tokenizer = None
        failure = None
        try:
            profile = _profile()
            counter, tokenizer = _tokenizer(tokenizer_cache)
        except Exception as error:
            failure = _failure(error)
        for count in CASE_COUNTS:
            row = dict(case_id=f"authored-functions-{count}", function_count=count,
                status="not_attempted", failure=None, cycle=None, cleanup=None)
            if failure is None:
                with _world_private_fixture(scratch_root) as (fixture, cleanup):
                    cleanup["private_fixture_native_contexts_closed"] = False
                    try:
                        row["cycle"] = _cycle(fixture, count, counter)
                        cleanup["private_fixture_native_contexts_closed"] = True
                        row["status"] = "completed"
                    except Exception as error:
                        failure = row["failure"] = _failure(error)
                        row["status"] = "failed"
                row["cleanup"] = cleanup
                if not cleanup["private_fixture_cleanup_complete"] and failure is None:
                    failure = row["failure"] = dict(stage="trusted_worker_packet_fixture", error_type="ValueError",
                        reason_code="private_fixture_cleanup_incomplete")
                    row["status"] = "failed"
            cases.append(row)
        unchanged = before == _pins()
        cleaned = all(row["cleanup"] is not None and row["cleanup"]["private_fixture_cleanup_complete"] for row in cases)
        qualified = failure is None and unchanged and cleaned and all(row["status"] == "completed" for row in cases)
        dispositions = {name: sum(row["cycle"] is not None and row["cycle"]["measurement"]["comparison"]["offline_comparison_disposition"] == name for row in cases)
            for name in ("compiled_smaller", "raw_comparison_preferred_not_smaller", "token_count_unavailable")}
        report = dict(schema=SCHEMA, scope=SCOPE, qualified=qualified, source_pins=before,
            source_pins_unchanged=unchanged, profile=profile, tokenizer=tokenizer, cases=cases,
            failure=failure, private_fixture_cleanup_complete=cleaned, offline_comparison_counts=dispositions,
            storage_scope="three_sequential_private_fixtures_each_bounded_and_cleaned_before_next",
            inner_resource_accounting_scope="container_local_fixture_authority_only",
            native_transport="authored_FakeQuackTransport_no_actual_TCP",
            provider_usage=dict(complete_native_total_tokens=None, billed_tokens=None, provider_session_population_complete=None),
            **FALSE)
        report["report_sha256"] = _sha(_wire(report))
        _bounded(report)
        _need(len(_wire(report)) <= MAX_REPORT_BYTES, "worker_packet_report_bound")
        return report
    finally:
        os.chdir(previous); _LOCK.release()


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, low=0, high=MAX_COUNTER):
    return type(value) is int and low <= value <= high


def _validate_measurement(value):
    _need(type(value) is dict and set(value) == {"bytes", "sha256", "local_o200k_tokens"}, "measurement_fields_differ")
    _need(_integer(value["bytes"], high=MAX_TEXT_BYTES) and _hash(value["sha256"])
        and (value["local_o200k_tokens"] is None or _integer(value["local_o200k_tokens"])), "measurement_invalid")


def _validate_comparison(value):
    fields = {"measurements", "offline_comparison_disposition", "service_delivery_mode", "actual_model_route_fallback",
        "trace_scope", "source_bytes_already_in_initial_prompt", "ipc_frames_added_to_model_tokens",
        "protocol_displayed_to_model", "complete_native_total_tokens", "billed_tokens"}
    _need(type(value) is dict and set(value) == fields, "comparison_fields_differ")
    metrics = value["measurements"]
    _need(type(metrics) is dict and set(metrics) == {"baseline_initial", "candidate_initial", "common_authored_protocol_text",
        "baseline_authored_trace", "candidate_authored_trace"}, "comparison_population_differs")
    for item in metrics.values(): _validate_measurement(item)
    for arm in ("baseline", "candidate"):
        _need(metrics[arm + "_authored_trace"]["bytes"] == metrics[arm + "_initial"]["bytes"] + metrics["common_authored_protocol_text"]["bytes"], "trace_bytes_differ")
    tokens = [item["local_o200k_tokens"] for item in metrics.values()]
    _need(all(item is None for item in tokens) or all(type(item) is int for item in tokens), "tokenizer_population_incomplete")
    old, new = metrics["baseline_initial"], metrics["candidate_initial"]
    disposition = ("token_count_unavailable" if old["local_o200k_tokens"] is None else "compiled_smaller"
        if new["bytes"] < old["bytes"] and new["local_o200k_tokens"] < old["local_o200k_tokens"] else "raw_comparison_preferred_not_smaller")
    _need(value["offline_comparison_disposition"] == disposition and value["service_delivery_mode"] == "compiled_packet_only"
        and value["actual_model_route_fallback"] is False and value["source_bytes_already_in_initial_prompt"] is True
        and value["ipc_frames_added_to_model_tokens"] is False and value["protocol_displayed_to_model"] is None
        and value["complete_native_total_tokens"] is None and value["billed_tokens"] is None
        and value["trace_scope"] == "one_authored_initial_prompt_plus_same_discovery_guidance_and_helper_receipts_once", "comparison_scope_differs")


def _validate_cycle(cycle, count, tokenizer):
    _need(type(cycle) is dict and set(cycle) == {"checks", "measurement"} and type(cycle["checks"]) is dict
        and set(cycle["checks"]) == CHECKS and all(value is True for value in cycle["checks"].values()), "cycle_checks_differ")
    value = cycle["measurement"]
    hashed = {"source_sha256", "task_projection_sha256", "binding_metadata_sha256", "positive_receipt_sha256", "negative_receipt_sha256"}
    fields = hashed | {"selected_source_files", "selected_source_bytes", "authored_function_count", "required_reference_count", "structural_reference_count",
        "native_validation_count", "native_validation_coverage", "packet_text", "required_source_text", "worker_peer", "ipc", "comparison"}
    _need(type(value) is dict and set(value) == fields and all(_hash(value[name]) for name in hashed), "cycle_fields_differ")
    _need(value["source_sha256"] == _sha(_source(count)) and value["selected_source_files"] == 1
        and type(value["selected_source_files"]) is int and value["selected_source_bytes"] == len(_source(count))
        and type(value["selected_source_bytes"]) is int and value["authored_function_count"] == count
        and type(value["authored_function_count"]) is int and value["required_reference_count"] == 5
        and type(value["required_reference_count"]) is int and _integer(value["structural_reference_count"], high=8)
        and type(value["native_validation_count"]) is int and value["native_validation_count"] == 0
        and value["native_validation_coverage"] == "no_native_validation_commands_declared", "source_or_native_population_differs")
    for name in ("packet_text", "required_source_text"): _validate_measurement(value[name])
    _need(value["required_source_text"]["sha256"] == value["source_sha256"]
        and value["required_source_text"]["bytes"] == value["selected_source_bytes"], "required_source_measurement_differs")
    peer = value["worker_peer"]
    _need(type(peer) is dict and set(peer) == {"pid", "uid", "gid", "start_time_ticks", "root_pid", "root_start_time_ticks", "groups_sha256", "group_count"}
        and _integer(peer["uid"]) and peer["uid"] == 1001 and _integer(peer["gid"]) and peer["gid"] == 1000
        and all(_integer(peer[key], low=1) for key in ("pid", "root_pid", "start_time_ticks", "root_start_time_ticks"))
        and _integer(peer["group_count"], low=1, high=64) and _hash(peer["groups_sha256"]), "worker_peer_differs")
    ipc = value["ipc"]
    fields = {"accepted_connections", "sent_responses", "refused_responses", "sent_packet_responses", "acknowledged_packets", "sent_packet_bytes",
        "sent_response_bytes", "sent_response_wire_bytes", "helper_stdin_bytes", "helper_stdout_bytes", "packet_body_counted_once", "byte_scope"}
    _need(type(ipc) is dict and set(ipc) == fields and all(_integer(ipc[key]) for key in fields - {"packet_body_counted_once", "byte_scope"}), "ipc_counter_invalid")
    _need(ipc["accepted_connections"] == ipc["sent_responses"] == 3 and ipc["refused_responses"] == ipc["sent_packet_responses"] == ipc["acknowledged_packets"] == 1
        and ipc["sent_packet_bytes"] == value["packet_text"]["bytes"] and ipc["sent_response_bytes"] >= ipc["sent_packet_bytes"]
        and ipc["sent_response_wire_bytes"] == ipc["sent_response_bytes"] + 12 and ipc["packet_body_counted_once"] is True
        and ipc["byte_scope"] == "actual_service_canonical_response_frames_and_exact_helper_transcript", "ipc_population_differs")
    _validate_comparison(value["comparison"])
    _need(tokenizer is not None and all((item["local_o200k_tokens"] is not None) is tokenizer["available"]
        for item in (value["packet_text"], value["required_source_text"], *value["comparison"]["measurements"].values())), "tokenizer_availability_differs")


def _validate_cleanup(value):
    required = {"private_fixture_cleanup_complete", "private_fixture_files", "private_fixture_bytes", "private_fixture_storage_bound_bytes",
        "private_fixture_cleanup_directories_made_owner_writable", "private_fixture_cleanup_scope", "private_fixture_root_removed",
        "private_fixture_cleanup_removed_files", "private_fixture_cleanup_removed_directories", "private_fixture_native_contexts_closed"}
    optional = {"private_fixture_observed_prefix_entries", "private_fixture_observed_prefix_files", "private_fixture_observed_prefix_bytes", "private_fixture_cleanup_reason"}
    _need(type(value) is dict and required <= set(value) <= required | optional, "cleanup_fields_differ")
    _need(value["private_fixture_cleanup_scope"] == CLEANUP_SCOPE and value["private_fixture_root_removed"] is False
        and type(value["private_fixture_storage_bound_bytes"]) is int and value["private_fixture_storage_bound_bytes"] == MAX_FIXTURE_BYTES, "cleanup_scope_differs")
    for name in ("private_fixture_cleanup_complete", "private_fixture_native_contexts_closed"):
        _need(type(value[name]) is bool, "cleanup_boolean_invalid")
    for name in required - {"private_fixture_cleanup_scope", "private_fixture_root_removed", "private_fixture_cleanup_complete", "private_fixture_native_contexts_closed"}:
        unknown = name in {"private_fixture_files", "private_fixture_bytes"} and value[name] is None
        _need(unknown and value["private_fixture_cleanup_complete"] is False
            or _integer(value[name], high=MAX_FIXTURE_BYTES), "cleanup_counter_invalid")
    _need((value["private_fixture_files"] is None) is (value["private_fixture_bytes"] is None), "cleanup_census_presence_differs")
    for name in optional - {"private_fixture_cleanup_reason"}:
        if name in value: _need(_integer(value[name], high=MAX_FIXTURE_BYTES), "cleanup_counter_invalid")
    if "private_fixture_cleanup_reason" in value:
        _need(type(value["private_fixture_cleanup_reason"]) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", value["private_fixture_cleanup_reason"]), "cleanup_reason_invalid")
    _need(not value["private_fixture_cleanup_complete"] or value["private_fixture_native_contexts_closed"], "cleanup_live_context_claim")


def _validate_failure(value):
    _need(value is None or type(value) is dict and set(value) == {"stage", "error_type", "reason_code"}
        and value["stage"] == "trusted_worker_packet_fixture" and type(value["error_type"]) is str
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value["error_type"]) and type(value["reason_code"]) is str
        and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", value["reason_code"]), "failure_fields_invalid")


def validate_report(report, *, expected_report_sha256):
    """Validate a caller-pinned observation; this does not replay native work."""
    _bounded(report)
    fields = {"schema", "scope", "qualified", "source_pins", "source_pins_unchanged", "profile", "tokenizer", "cases", "failure",
        "private_fixture_cleanup_complete", "offline_comparison_counts", "storage_scope", "inner_resource_accounting_scope", "native_transport", "provider_usage", "report_sha256", *FALSE}
    _need(type(report) is dict and set(report) == fields and len(_wire(report)) <= MAX_REPORT_BYTES, "packet_report_fields_or_bound")
    _need(_hash(expected_report_sha256) and report_sha256(report) == expected_report_sha256, "independent_packet_report_pin_differs")
    body = dict(report); seal = body.pop("report_sha256")
    _need(seal == _sha(_wire(body)), "packet_report_self_seal_differs")
    _need(report["schema"] == SCHEMA and report["scope"] == SCOPE and report["native_transport"] == "authored_FakeQuackTransport_no_actual_TCP"
        and report["inner_resource_accounting_scope"] == "container_local_fixture_authority_only"
        and report["storage_scope"] == "three_sequential_private_fixtures_each_bounded_and_cleaned_before_next", "packet_report_scope_differs")
    for key, expected in FALSE.items():
        _need(type(report[key]) is type(expected) and report[key] == expected, "packet_report_authority_claim")
    for key in ("qualified", "source_pins_unchanged", "private_fixture_cleanup_complete"):
        _need(type(report[key]) is bool, "packet_report_boolean_invalid")
    _need(report["provider_usage"] == dict(complete_native_total_tokens=None, billed_tokens=None, provider_session_population_complete=None), "provider_usage_claim")
    pins = report["source_pins"]
    _need(type(pins) is dict and set(pins) == {"scope", "files"} and pins["scope"] == "listed_source_files_only_not_transitive_provenance"
        and type(pins["files"]) is dict and set(pins["files"]) == set(SOURCE_PATHS), "source_pin_population_differs")
    for pin in pins["files"].values():
        _need(type(pin) is dict and set(pin) == {"bytes", "sha256"} and _integer(pin["bytes"], high=2 * 1024**2) and _hash(pin["sha256"]), "source_pin_invalid")
    profile = report["profile"]
    if profile is not None:
        _need(type(profile) is dict and set(profile) == {"owner_uid", "owner_gid", "worker_uid", "worker_gid", "launcher_sha256", "literal_popen_argv_sha256", "worker_peer_argv_sha256", "namespaces"}, "profile_fields_differ")
        _need(all(_integer(profile[key]) and profile[key] == expected for key, expected in (("owner_uid",1000),("owner_gid",1000),("worker_uid",1001),("worker_gid",1000)))
            and profile["launcher_sha256"] == _sha(WORKER_ENTRY_SOURCE.encode()) and profile["literal_popen_argv_sha256"] == _sha(_wire(list(WORKER_ARGV)))
            and profile["worker_peer_argv_sha256"] == _sha(_wire(list(WORKER_PEER_ARGV))), "profile_identity_differs")
        _need(type(profile["namespaces"]) is dict and set(profile["namespaces"]) == {"pid","mnt","net"}
            and all(type(value) is str and re.fullmatch(name + r":\[[0-9]+\]", value) for name, value in profile["namespaces"].items()), "profile_namespace_invalid")
    tokenizer = report["tokenizer"]
    if tokenizer is not None:
        _need(type(tokenizer) is dict and set(tokenizer) == {"available", "encoding", "ranks_sha256", "reason", "count_scope", "provider_compatibility_verified", "provider_usage", "message_framing_included", "download_performed"}
            and type(tokenizer["available"]) is bool and tokenizer["count_scope"] == "exact_local_encoding_of_authored_text_only"
            and all(tokenizer[key] is False for key in ("provider_compatibility_verified", "provider_usage", "message_framing_included", "download_performed")), "tokenizer_scope_invalid")
        _need(tokenizer["encoding"] == ("o200k_base" if tokenizer["available"] else None)
            and (tokenizer["ranks_sha256"] == TOKENIZER_RANKS_SHA256 if tokenizer["available"] else tokenizer["ranks_sha256"] is None)
            and (tokenizer["reason"] is None if tokenizer["available"] else type(tokenizer["reason"]) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", tokenizer["reason"])), "tokenizer_identity_invalid")
    _validate_failure(report["failure"])
    _need(type(report["cases"]) is list and len(report["cases"]) == len(CASE_COUNTS), "case_population_differs")
    dispositions = dict.fromkeys(("compiled_smaller", "raw_comparison_preferred_not_smaller", "token_count_unavailable"), 0)
    stopped = False
    for row, count in zip(report["cases"], CASE_COUNTS):
        _need(type(row) is dict and set(row) == {"case_id", "function_count", "status", "failure", "cycle", "cleanup"}
            and row["case_id"] == f"authored-functions-{count}" and type(row["function_count"]) is int and row["function_count"] == count
            and row["status"] in {"completed", "failed", "not_attempted"}, "case_identity_differs")
        _validate_failure(row["failure"])
        _need(not stopped or row["status"] == "not_attempted", "case_execution_after_failure")
        if row["status"] == "not_attempted":
            _need(row["cycle"] is row["cleanup"] is row["failure"] is None and report["failure"] is not None, "not_attempted_case_invalid")
            stopped = True
        else:
            _validate_cleanup(row["cleanup"])
            _need(profile is not None and tokenizer is not None, "attempted_case_profile_missing")
            if row["cycle"] is not None:
                _validate_cycle(row["cycle"], count, tokenizer)
                dispositions[row["cycle"]["measurement"]["comparison"]["offline_comparison_disposition"]] += 1
            if row["status"] == "completed":
                _need(row["cycle"] is not None and row["failure"] is None and row["cleanup"]["private_fixture_cleanup_complete"], "completed_case_invalid")
            else:
                _need(row["failure"] is not None and row["failure"] == report["failure"], "failed_case_invalid")
                stopped = True
    _need(type(report["offline_comparison_counts"]) is dict and report["offline_comparison_counts"] == dispositions
        and all(type(value) is int for value in report["offline_comparison_counts"].values()), "comparison_denominator_differs")
    cleaned = all(row["cleanup"] is not None and row["cleanup"]["private_fixture_cleanup_complete"] for row in report["cases"])
    qualified = report["failure"] is None and report["source_pins_unchanged"] and cleaned and all(row["status"] == "completed" for row in report["cases"])
    _need(report["private_fixture_cleanup_complete"] is cleaned and report["qualified"] is qualified, "report_qualification_differs")
    return dict(valid=True, qualified=qualified, report_sha256=expected_report_sha256,
        native_operations_independently_replayed=False, **FALSE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_experiment(scratch_root=arguments.scratch_root, tokenizer_cache=arguments.tokenizer_cache)
    validate_report(report, expected_report_sha256=report_sha256(report))
    with arguments.output.open("xb") as stream: stream.write(_wire(report) + b"\n")
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
