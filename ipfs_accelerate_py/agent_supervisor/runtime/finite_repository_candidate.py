"""Trusted bounded subprocess proposals for one reviewed finite source change.

This additive prerequisite never edits canonical source, starts Portal, grants
completion, or isolates an untrusted worker. The signed finite admission keeps
its complete administrator task population. A same-UID, fixed trusted Python
child only reproduces the independently declared replacement in a private
workspace; its process receipt is retained evidence, not origin attestation.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from ipfs_datasets_py.logic.backends import process as native_process
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as observation
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
    IntegerOffsetContract, compile_integer_offset,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as native_resources

from ..planning import finite_integer_capacity as capacity
from ..planning import finite_integer_codebase as matcher
from ..planning.finite_integer_source_custody import capture_source_custody
from ..planning import finite_integer_source_custody as source_module
from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ..prompt.prompt_workflow import PromptGoalGraph
from . import finite_repository_admission as finite
from . import local_planning_admission as local

PROFILE = "finite-repository-trusted-offset-candidate@1"
CANDIDATE_SCHEMA = "finite-repository-reviewed-candidate@1"
RESULT_SCHEMA = "finite-repository-generated-candidate@1"
BEFORE_SOURCE = b"def increment(n: int) -> int:\n    return n + 1\n"
AFTER_SOURCE = b"def increment(n: int) -> int:\n    return n + 2\n"
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority",
    "publication_authority", "production_activation", "native_worker_loop_qualified",
    "untrusted_worker_isolated", "filesystem_isolation", "network_isolation",
    "owner_keys_inaccessible", "process_origin_attested", "convergence_proved",
)}
_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
_LIMITS = {"child_timeout_ms": 5000, "cpu_seconds": 5,
    "address_space_bytes": 512 * 1024**2, "resident_memory_bytes": 512 * 1024**2,
    "reservation_memory_mb": 1024, "child_memory_mb": 512,
    "cpu_slots": 1, "child_process_slots": 1,
    "max_input_bytes": 256 * 1024, "max_output_bytes": 64 * 1024,
    "max_workspace_bytes": 1024 * 1024, "max_output_files": 8,
    "memory_control": "Per-process address-space cap and sampled process-tree RSS guard; no aggregate kernel cgroup ceiling."}
_DRIVER = ("""import base64,hashlib,json,os,pathlib,sys
before=""" + repr(BEFORE_SOURCE) + "\nafter=" + repr(AFTER_SOURCE) + """
if len(sys.argv)!=2: raise SystemExit('fixed candidate digest required')
raw=pathlib.Path('candidate.json').read_bytes()
if hashlib.sha256(raw).hexdigest()!=sys.argv[1]: raise SystemExit('candidate envelope changed')
envelope=json.loads(raw)
if set(envelope)!={'payload','binding'}: raise SystemExit('closed envelope required')
p=envelope['payload']
if pathlib.Path('source.py').read_bytes()!=before: raise SystemExit('preimage changed')
if base64.b64decode(p['before_base64'],validate=True)!=before or base64.b64decode(p['after_base64'],validate=True)!=after: raise SystemExit('exact reviewed bytes required')
if p['before_sha256']!=hashlib.sha256(before).hexdigest() or p['after_sha256']!=hashlib.sha256(after).hexdigest(): raise SystemExit('source binding differs')
if p['path']!='calc.py' or p['function_name']!='increment' or p['parameter']!='n' or type(p['desired_offset']) is not int or p['desired_offset']!=2: raise SystemExit('reviewed target differs')
pathlib.Path('candidate.py').write_bytes(after)
receipt={'schema':'finite-repository-trusted-child-receipt@1','candidate_sha256':sys.argv[1],'task_cid':p['task_cid'],'before_sha256':p['before_sha256'],'after_sha256':p['after_sha256'],'pid':os.getpid(),'uid':os.getuid(),'provider_calls':0,'publication_authority':False,'completion_authority':False,'environment_keys':sorted(os.environ)}
pathlib.Path('child-receipt.json').write_text(json.dumps(receipt,sort_keys=True,separators=(',',':')))
""").encode()
_ARTIFACT_NAMES = {"candidate": "candidate.json", "source": "source.py", "driver": "driver.py",
    "replacement": "candidate.py", "child_receipt": "child-receipt.json",
    "process": "process.json", "current_verification": "current-verification.json",
    "source_custody": "source-custody.json"}
_RESULT_FIELDS = {"schema", "profile", "status", "parent_admission", "parent_admission_cid",
    "reviewed_candidate", "reviewed_candidate_cid", "head", "task_cid", "source_custody",
    "source_cid", "replacement_cid", "canonical_source_unchanged", "proposal_generated",
    "provider_calls", "training_steps", "observed_current_on_return", "reservation_released_on_return",
    "native_capacity", "child_process", "artifacts", "output", "elapsed_ms", "result_cid", *_FALSE}
_PROCESS_FIELDS = {"interface_version", "runtime", "command", "returncode", "stdout", "stderr", "pid",
    "timed_out", "cancelled", "unavailable", "output_truncated", "workspace_limit_exceeded",
    "process_tree_terminated", "resource_exhausted", "workspace_cleaned", "termination_reason", "error",
    "elapsed_ms", "effective_timeout_ms", "limits"}
_SHA = re.compile(r"[0-9a-f]{64}")
_LANE = native_resources.ResourceLane.VALIDATION.value


class FiniteRepositoryCandidateError(ValueError):
    """A reviewed binding, owned process or retained artifact was refused."""


def _need(value, message):
    if not value:
        raise FiniteRepositoryCandidateError(message)


def _plain(value):
    try:
        raw = canonical_dag_json_bytes(value)
        _need(len(raw) <= finite.MAX_BYTES, "bounded canonical candidate record required")
        return json.loads(raw)
    except (TypeError, ValueError, RecursionError) as error:
        if isinstance(error, FiniteRepositoryCandidateError):
            raise
        raise FiniteRepositoryCandidateError("exact bounded canonical candidate values required") from error


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _same(left, right):
    return canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)


def _decode_native(value):
    if type(value) is dict and set(value) == {"native_float_hex"}:
        _need(type(value["native_float_hex"]) is str, "exact native float encoding required")
        try:
            decoded = float.fromhex(value["native_float_hex"])
        except ValueError as error:
            raise FiniteRepositoryCandidateError("malformed native float encoding") from error
        _need(math.isfinite(decoded) and decoded.hex() == value["native_float_hex"],
              "canonical finite native float encoding required")
        return decoded
    if type(value) is dict:
        return {key: _decode_native(row) for key, row in value.items()}
    if type(value) is list:
        return [_decode_native(row) for row in value]
    return value


def _verify_capacity(value, child_pid):
    _need(type(value) is dict and set(value) == {"configuration", "observations", "authority_scope"}
          and value["authority_scope"] == "reservation held only during trusted proposal subprocess",
          "closed native candidate capacity record required")
    configured = value["configuration"]
    _need(type(configured) is dict and set(configured) == {"persisted", "lease_ttl_seconds", "auto_renew_leases", "state_path"}
          and type(configured["auto_renew_leases"]) is bool
          and type(configured["state_path"]) is str and Path(configured["state_path"]).is_absolute(),
          "complete exact native candidate capacity configuration required")
    decoded = _decode_native(configured)
    _need(type(decoded["lease_ttl_seconds"]) is float and decoded["lease_ttl_seconds"] > 0,
          "native lease expiry policy required")
    configuration = native_resources.ResourceSchedulerConfig(**decoded["persisted"],
        lease_ttl_seconds=decoded["lease_ttl_seconds"], auto_renew_leases=decoded["auto_renew_leases"],
        state_path=decoded["state_path"])
    configuration.validate()
    _need(configuration.proof_safety_enabled is True and _same(configured, capacity._inert_native({
        "persisted": configuration.persisted_dict(), "lease_ttl_seconds": float(configuration.lease_ttl_seconds),
        "auto_renew_leases": configuration.auto_renew_leases, "state_path": str(configuration.state_path)})),
        "native proof-safety configuration cannot be reconstructed")
    rows = value["observations"]
    _need(type(rows) is list and len(rows) == 4, "exact before/after root and child observations required")
    reservations, measured = [], []
    for row, memory in zip(rows, (1024, 512, 512, 1024)):
        _need(type(row) is dict and set(row) == {"measured_at_ms", "reservation", "host", "reserved_cpu_slots", "reserved_memory_mb"}
              and type(row["measured_at_ms"]) is int and row["measured_at_ms"] > 0
              and type(row["reserved_cpu_slots"]) is int and row["reserved_cpu_slots"] >= 1
              and type(row["reserved_memory_mb"]) is int and row["reserved_memory_mb"] >= 1024,
              "exact bounded native capacity observation required")
        r = _decode_native(row["reservation"])
        _need(type(r) is dict and set(r) == {"lease_id", "parent_lease_id", "owner_pid", "owner_birth_marker", "lane",
              "cpu_slots", "memory_mb", "child_process_slots", "requires_gpu", "expires_at"}
              and type(r["lease_id"]) is str and bool(r["lease_id"])
              and (r["parent_lease_id"] is None or type(r["parent_lease_id"]) is str)
              and type(r["owner_pid"]) is int and r["owner_pid"] > 0 and r["owner_pid"] != child_pid
              and type(r["owner_birth_marker"]) is str and bool(r["owner_birth_marker"])
              and r["lane"] == _LANE and type(r["cpu_slots"]) is int and r["cpu_slots"] == 1
              and type(r["memory_mb"]) is int and r["memory_mb"] == memory
              and type(r["child_process_slots"]) is int and r["child_process_slots"] == 1
              and r["requires_gpu"] is False and type(r["expires_at"]) is float
              and r["expires_at"] * 1000 >= row["measured_at_ms"],
              "exact current owned native candidate reservation required")
        host = capacity.ProofHostResources(**_decode_native(row["host"]))
        _need(_same(row["host"], capacity._inert_native(asdict(host)))
              and host.memory_stall_percent < configuration.proof_memory_stall_percent
              and host.cpu_stall_percent < configuration.proof_cpu_stall_percent
              and host.io_stall_percent < configuration.proof_io_stall_percent
              and row["reserved_cpu_slots"] <= configuration.total_cpu_slots
              and row["reserved_memory_mb"] <= configuration.total_memory_mb,
              "retained actual native host capacity or pressure policy differs")
        reservations.append(r)
        measured.append(row["measured_at_ms"])
    _need(measured == sorted(measured), "native capacity observation order differs")
    for left, right in ((reservations[0], reservations[3]), (reservations[1], reservations[2])):
        _need(_same({k: v for k, v in left.items() if k != "expires_at"},
                    {k: v for k, v in right.items() if k != "expires_at"}),
              "native before/after reservation identity changed")
    _need(reservations[1]["parent_lease_id"] == reservations[0]["lease_id"]
          and reservations[1]["lease_id"] != reservations[0]["lease_id"]
          and reservations[1]["owner_pid"] == reservations[0]["owner_pid"],
          "native child reservation is not bound to the recorded owner root")


def _verify_custody(value, admission):
    original = admission["evidence"]["source_custody"]
    _need(type(value) is dict and set(value) == set(original)
          and value.get("schema") == source_module.SCHEMA
          and value.get("custody_cid") == cid_for_structured({k: v for k, v in value.items() if k != "custody_cid"})
          and all(value.get(name) is False for name in source_module._FALSE),
          "closed native source custody projection required")
    variable = {"custody_cid", "files", "working_directories", "git_directories"}
    _need(_same({k: v for k, v in value.items() if k not in variable},
                {k: v for k, v in original.items() if k not in variable}),
          "custody source/head/manifest/AST/staged projection differs from signed parent")
    for name in ("working_directories", "git_directories"):
        rows = value[name]
        _need(type(rows) is list and len(rows) == len(original[name])
              and [r["path"] for r in rows] == [r["path"] for r in original[name]],
              "custody complete directory population differs")
        for row, before in zip(rows, original[name]):
            _need(type(row) is dict and set(row) == {"path", "physical_identity"}
                  and type(row["physical_identity"]) is list
                  and len(row["physical_identity"]) == len(before["physical_identity"])
                  and all(type(v) is int and v >= 0 for v in row["physical_identity"]),
                  "custody exact physical directory observation required")
    rows = value["files"]
    _need(type(rows) is list and len(rows) == len(original["files"])
          and [r["role"] for r in rows] == [r["role"] for r in original["files"]],
          "custody complete source/CAS/AST/Git file population differs")
    for row, before in zip(rows, original["files"]):
        _need(type(row) is dict and set(row) == {"path", "role", "size_bytes", "sha256", "initial_read_identity", "validation"}
              and type(row["size_bytes"]) is int and 0 <= row["size_bytes"] <= source_module.LIMITS["native_object_bytes"]
              and type(row["sha256"]) is str and _SHA.fullmatch(row["sha256"]) is not None
              and type(row["initial_read_identity"]) is list and len(row["initial_read_identity"]) == 6
              and all(type(v) is int and v >= 0 for v in row["initial_read_identity"])
              and stat.S_ISREG(row["initial_read_identity"][2])
              and row["initial_read_identity"][3] == row["size_bytes"],
              "custody exact bounded physical file observation required")
        volatile = {"initial_read_identity"}
        if row["role"] == "git:index":
            volatile |= {"sha256", "size_bytes"}
        _need(_same({k: v for k, v in row.items() if k not in volatile},
                    {k: v for k, v in before.items() if k not in volatile})
              and row["initial_read_identity"][2] == before["initial_read_identity"][2],
              "custody frozen file identity differs from signed parent")


def _read(path, maximum, checkpoint=lambda: None):
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
          "canonical regular candidate artifact required")
    for parent in path.parents:
        _need(not parent.is_symlink(), "candidate artifact ancestor changed")
    checkpoint()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum,
              "candidate artifact byte bound exceeded")
        parts, size = [], 0
        while block := stream.read(64 * 1024):
            checkpoint()
            size += len(block)
            _need(size <= maximum, "candidate artifact grew beyond byte bound")
            parts.append(block)
        after, current = os.fstat(stream.fileno()), path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    _need(identity(before) == identity(after) == identity(current), "candidate artifact changed during read")
    checkpoint()
    return b"".join(parts)


def _pin(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": _sha(raw), "cid": cid_for_bytes(raw)}


def _implementation(checkpoint=lambda: None):
    rows = {}
    for module in (finite, local, capacity, source_module, observation, native_process, native_resources):
        path = Path(module.__file__).resolve(strict=True)
        raw = _read(path, 8 * 1024**2, checkpoint)
        rows[module.__name__] = _pin(path, raw)
    path = Path(__file__).resolve(strict=True)
    rows[__name__] = _pin(path, _read(path, 8 * 1024**2, checkpoint))
    return {"schema": "finite-repository-candidate-implementation@1", "source_pins": rows,
        "driver_sha256": _sha(_DRIVER), "driver_bytes": len(_DRIVER),
        "scope": "Selected trusted producer bytes and selected Python ELF; no transitive environment or process-origin attestation."}


def _review_ref(value):
    _need(type(value) is str and value == value.strip() and 1 <= len(value.encode()) <= 1024
          and "\0" not in value, "explicit bounded administrator review reference required")
    return value


def _expected(admission, review_ref, checkpoint=lambda: None):
    admission = _plain(admission)
    verified = finite.verify_finite_repository_admission(admission=admission)
    semantic = verified["semantic_context"]
    payload = admission["declaration"]["payload"]
    _need(semantic["residual_requirement_ids"] == [matcher.OFFSET_STATEMENT_ID]
          and semantic["eligible_requirement_ids"] == [matcher.TYPE_STATEMENT_ID]
          and verified["receipt"]["planning_permitted"] is True
          and admission["local_admission"] is not None,
          "exact one residual offset and complete original task population required")
    contract = IntegerOffsetContract.from_dict(semantic["query"]["contract"])
    _need(contract.to_dict() == IntegerOffsetContract(path="calc.py", function_name="increment",
        parameter="n", offset=2).to_dict(), "exact reviewed calc.py integer offset contract required")
    before = finite._artifacts(admission["evidence"]["match"]["observation"], checkpoint)["source"]
    _need(before == BEFORE_SOURCE, "exact independently reviewed n plus one preimage required")
    before_model = compile_integer_offset(before, contract, revision="snapshot:" + semantic["head"]["snapshot_cid"])
    after_model = compile_integer_offset(AFTER_SOURCE, contract, revision="reviewed-candidate:" + cid_for_bytes(AFTER_SOURCE))
    _need(before_model.body_offset == 1 and after_model.body_offset == 2,
          "independent guarded source projections differ from reviewed replacement")
    native = semantic["native_task_bindings"][matcher.OFFSET_STATEMENT_ID]
    graph = PromptGoalGraph.from_dict(admission["graph"])
    task = next((row for row in graph.tasks if row.task_cid == native["task_cid"]), None)
    _need(task is not None and task.task_key == native["task_key"] and len(task.outputs) == 1
          and task.outputs[0].path == contract.path and task.outputs[0].effect == "modify",
          "exact original residual task and one reviewed modification required")
    tool = payload["tool_policy"]["python"]
    _need(observation._tool(Path(tool["path"]), checkpoint) == tool,
          "selected original native Python ELF changed")
    return _plain({"schema": CANDIDATE_SCHEMA, "profile": PROFILE,
        "review_ref": _review_ref(review_ref), "review_origin": "existing_profile_owner_declaration_only",
        "parent_admission_cid": cid_for_structured(admission), "declaration_cid": cid_for_structured(admission["declaration"]),
        "graph_cid": cid_for_structured(admission["graph"]), "head": semantic["head"],
        "source_cid": semantic["source_cid"], "prompt_cid": payload["request"]["prompt_source_cid"],
        "intent_json_sha256": _sha(payload["intent_json"].encode()),
        "operation_catalog_cid": semantic["operation_catalog_cid"],
        "task_cid": task.task_cid, "task_key": task.task_key, "task": task.to_dict(),
        "administrator_task_cids": semantic["administrator_task_cids"],
        "path": contract.path, "function_name": contract.function_name, "parameter": contract.parameter,
        "desired_offset": contract.offset, "contract": contract.to_dict(), "contract_cid": contract.cid,
        "before_base64": base64.b64encode(BEFORE_SOURCE).decode(), "before_sha256": _sha(BEFORE_SOURCE),
        "after_base64": base64.b64encode(AFTER_SOURCE).decode(), "after_sha256": _sha(AFTER_SOURCE),
        "after_cid": cid_for_bytes(AFTER_SOURCE), "before_compiled_cid": before_model.cid,
        "after_compiled_cid": after_model.cid, "python": tool, "environment": _ENV,
        "resource_helper": observation._tool(Path(native_process._linux_prlimit_path()), checkpoint),
        "limits": _LIMITS, "implementation": _implementation(checkpoint),
        "child_trust": "fixed_trusted_same_uid_proposal_subprocess", **_FALSE})


def verify_finite_repository_candidate(*, admission, candidate):
    """Verify signed historical bindings; no current-source or execution grant."""
    admission, candidate = _plain(admission), _plain(candidate)
    _need(type(candidate) is dict and set(candidate) == {"payload", "binding"}
          and type(candidate["payload"]) is dict, "closed signed candidate envelope required")
    profile = finite._profile(admission["declaration"]["payload"]["manifest"])
    signed = local._verify_signature(candidate, profile)
    _need(_same(signed, _expected(admission, signed.get("review_ref"))),
          "complete reviewed candidate differs from independently rebuilt parent bindings")
    return {"schema": "finite-repository-candidate-historical-verification@1",
        "candidate_cid": cid_for_structured(candidate), "parent_admission_cid": cid_for_structured(admission),
        "payload": signed, "observed_current": False, **_FALSE}


def _owner(owner, payload):
    _need(type(owner) is RepositoryPlanPreviewOwner and owner.memory_mb >= 1024,
          "exact native candidate owner with complete finite preparation memory required")
    _need(_same(owner.expected_head.to_dict(), payload["head"]), "candidate owner head differs")
    scheduler = capacity._native_scheduler(owner)
    capacity._config(scheduler)
    _need(owner.parent_lease is None or type(owner.parent_lease) is native_resources.ResourceLease,
          "actual native parent lease required; serialized identifiers are not authority")
    return scheduler


def author_finite_repository_candidate(*, owner, admission, review_ref):
    """Owner-sign fixed reviewed bytes without running a worker or modifying source."""
    admission = _plain(admission)
    payload = _expected(admission, review_ref)
    _owner(owner, payload)
    _need(str(owner.repository) == admission["declaration"]["payload"]["manifest"]["payload"]["repository"],
          "candidate owner repository differs")
    custody = capture_source_custody(owner)
    envelope = _plain(local._signed(payload, admission["declaration"]["payload"]["manifest"]["payload"]))
    verify_finite_repository_candidate(admission=admission, candidate=envelope)
    custody.require_current()
    return envelope


def _write(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return _pin(path, raw)


def _capacity_observation(scheduler, lease, request_id, checkpoint):
    checkpoint()
    with scheduler._locked_state(persist=False) as state:
        row = capacity._authenticate(scheduler, lease, state, time.time())
        _need(row["owner_pid"] == os.getpid() and row["request_id"] == request_id,
              "native candidate reservation owner or identity changed")
        host, reserved_cpu, reserved_memory = capacity._host(scheduler, state, time.time())
        clean = {key: row[key] for key in ("lease_id", "parent_lease_id", "owner_pid", "owner_birth_marker",
            "lane", "cpu_slots", "memory_mb", "child_process_slots", "requires_gpu", "expires_at")}
        _need(clean["cpu_slots"] == 1 and clean["memory_mb"] in {512, 1024}
              and clean["child_process_slots"] == 1 and clean["requires_gpu"] is False,
              "exact native candidate capacity required")
        result = capacity._inert_native({"measured_at_ms": math.ceil(time.time() * 1000),
            "reservation": clean, "host": asdict(host),
            "reserved_cpu_slots": reserved_cpu, "reserved_memory_mb": reserved_memory})
    checkpoint()
    return _plain(result)


def generate_finite_repository_candidate(*, owner, admission, candidate, output, policy_observer):
    """Fresh finite validation, actual bounded child, and detached return fences.

    Preparation and execution share the owner's at most 90-second deadline.
    The independently live child reservation has a fixed at most five-second
    execution cap; the old preview forecast cannot authorize this subprocess.
    """
    _need(type(owner) is RepositoryPlanPreviewOwner, "exact native candidate owner required")
    started = time.monotonic()
    deadline = started + owner.timeout_seconds
    scheduler = None
    def checkpoint():
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise native_resources.LeaseCancelledError("finite candidate cancelled")
        if scheduler is not None and owner.parent_lease is not None:
            try:
                with scheduler._locked_state(persist=False) as state:
                    capacity._authenticate(scheduler, owner.parent_lease, state, time.time())
            except capacity.FiniteIntegerCapacityError as error:
                raise native_resources.LeaseCancelledError("finite candidate parent authority was revoked") from error
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise native_resources.LeaseTimeoutError("finite candidate deadline exceeded")
        return duration
    checkpoint()
    admission, candidate = _plain(admission), _plain(candidate)
    verified = verify_finite_repository_candidate(admission=admission, candidate=candidate)
    payload = verified["payload"]
    scheduler = _owner(owner, payload)
    configuration = capacity._config(scheduler)
    _need(str(owner.repository) == admission["declaration"]["payload"]["manifest"]["payload"]["repository"]
          and callable(policy_observer), "exact owner repository and current policy observer required")
    output = Path(output)
    _need(output.is_absolute() and output.resolve() == output and not output.exists()
          and not output.is_relative_to(owner.repository) and output.parent.is_dir(),
          "fresh canonical candidate output outside repository required")
    deadline = min(deadline, started +
        admission["declaration"]["payload"]["request"]["budget"]["max_latency_ms"] / 1000)
    checkpoint()
    if owner.parent_lease is not None:
        with scheduler._locked_state(persist=False) as state:
            parent = capacity._authenticate(scheduler, owner.parent_lease, state, time.time())
            free = capacity._unspent(state, parent)
            _need(free["cpu_slots"] >= 2 and free["memory_mb"] >= owner.memory_mb * 2
                  and free["child_process_slots"] >= 2,
                  "parent must cover complete fresh finite preparation before child execution")
    custody = capture_source_custody(owner, checkpoint)
    output.mkdir(mode=0o700)
    current = finite.verify_current_finite_repository_admission(
        owner=replace(owner, timeout_seconds=checkpoint()), admission=admission,
        output=output / "current-preview", policy_observer=policy_observer)
    _need(current["observed_current"] is True and _same(current["semantic_context"],
          finite.verify_finite_repository_admission(admission=admission)["semantic_context"]),
          "fresh complete finite context differs from signed parent")
    proof_fence = finite._artifact_fence(owner, current["fresh_evidence"], checkpoint)
    parent_proof_fence = finite._artifact_fence(owner, admission["evidence"], checkpoint)
    raw_candidate = canonical_dag_json_bytes(candidate)
    artifacts = {}
    for role, raw in (("candidate", raw_candidate), ("source", BEFORE_SOURCE), ("driver", _DRIVER),
                      ("source_custody", canonical_dag_json_bytes(custody.material_binding)),
                      ("current_verification", canonical_dag_json_bytes(current))):
        artifacts[role] = _write(output / _ARTIFACT_NAMES[role], raw)
    expected_files = {row["path"]: raw for row, raw in (
        (artifacts["candidate"], raw_candidate), (artifacts["source"], BEFORE_SOURCE),
        (artifacts["source_custody"], canonical_dag_json_bytes(custody.material_binding)),
        (artifacts["driver"], _DRIVER), (artifacts["current_verification"], canonical_dag_json_bytes(current)))}
    def fence():
        checkpoint()
        _need(_same(capacity._config(scheduler), configuration), "native capacity configuration changed")
        _need(_same(_implementation(checkpoint), payload["implementation"]), "trusted implementation or driver changed")
        _need(observation._tool(Path(payload["python"]["path"]), checkpoint) == payload["python"],
              "selected native Python ELF changed")
        _need(_same(observation._tool(Path(native_process._linux_prlimit_path()), checkpoint), payload["resource_helper"]),
              "selected native resource helper ELF changed")
        parent_proof_fence()
        proof_fence()
        custody.require_current(checkpoint)
        for path, raw in expected_files.items():
            _need(_read(Path(path), finite.MAX_BYTES, checkpoint) == raw, "frozen candidate artifact changed")
        checkpoint()
    fence()
    request_id = "finite-candidate:" + verified["candidate_cid"]
    root_controls = dict(lane=native_resources.ResourceLane.VALIDATION, cpu_slots=1,
        memory_mb=1024, child_process_slots=1, timeout=min(30, checkpoint()),
        cancel_event=owner.cancel_event, request_id=request_id)
    lease = (owner.parent_lease.acquire_child(**root_controls) if owner.parent_lease is not None
             else scheduler.acquire(**root_controls))
    observations = []
    with lease:
        combined = lease.combined_cancellation_signal(owner.cancel_event)
        fence()
        observations.append(_capacity_observation(scheduler, lease, request_id, checkpoint))
        with lease.acquire_child(lane=native_resources.ResourceLane.VALIDATION, cpu_slots=1,
                memory_mb=512, child_process_slots=1, timeout=checkpoint(), cancel_event=combined,
                request_id=request_id + ":child") as child:
            observations.append(_capacity_observation(scheduler, child, request_id + ":child", checkpoint))
            child_limits = native_process.ToolRunLimits(timeout_seconds=min(5, checkpoint()), cpu_seconds=5,
                memory_bytes=_LIMITS["address_space_bytes"], resident_memory_bytes=_LIMITS["resident_memory_bytes"],
                max_input_bytes=_LIMITS["max_input_bytes"], max_output_bytes=_LIMITS["max_output_bytes"],
                max_workspace_bytes=_LIMITS["max_workspace_bytes"], max_output_files=_LIMITS["max_output_files"])
            raw = native_process.BoundedToolRunner(base_environment=_ENV).run(
                [payload["python"]["path"], "-I", "-S", "driver.py", _sha(raw_candidate)],
                input_files={"driver.py": _DRIVER, "source.py": BEFORE_SOURCE, "candidate.json": raw_candidate},
                output_paths=("candidate.py", "child-receipt.json"),
                limits=child_limits,
                cancellation=child.combined_cancellation_signal(combined))
            if combined.is_set() or raw.cancelled:
                raise native_resources.LeaseCancelledError("native candidate child cancelled")
            _need(raw.ok and raw.workspace_cleaned is True and not any(getattr(raw, name) for name in
                ("output_truncated", "workspace_limit_exceeded", "process_tree_terminated"))
                and set(raw.output_files) == {"candidate.py", "child-receipt.json"}
                and raw.output_files["candidate.py"] == AFTER_SOURCE and raw.stdout == raw.stderr == "",
                "bounded trusted candidate child did not produce the exact reviewed output")
            observations.append(_capacity_observation(scheduler, child, request_id + ":child", checkpoint))
            fence()
        observations.append(_capacity_observation(scheduler, lease, request_id, checkpoint))
        fence()
    process = raw.to_dict()
    process.pop("output_files")
    elapsed = process.pop("elapsed_seconds")
    process["elapsed_ms"] = max(0, int(elapsed * 1000))
    process["effective_timeout_ms"] = math.ceil(child_limits.timeout_seconds * 1000)
    process["limits"] = payload["limits"]
    for role, content in (("replacement", AFTER_SOURCE), ("child_receipt", raw.output_files["child-receipt.json"]),
                          ("process", canonical_dag_json_bytes(process))):
        artifacts[role] = _write(output / _ARTIFACT_NAMES[role], content)
        expected_files[artifacts[role]["path"]] = content
    record = _plain({"schema": RESULT_SCHEMA, "profile": PROFILE, "status": "candidate_generated",
        "parent_admission": admission, "parent_admission_cid": cid_for_structured(admission),
        "reviewed_candidate": candidate, "reviewed_candidate_cid": verified["candidate_cid"],
        "head": payload["head"], "task_cid": payload["task_cid"], "source_cid": payload["source_cid"],
        "replacement_cid": payload["after_cid"], "source_custody": custody.material_binding,
        "canonical_source_unchanged": True, "proposal_generated": True, "provider_calls": 0,
        "training_steps": 0, "observed_current_on_return": True, "reservation_released_on_return": True,
        "native_capacity": {"configuration": configuration, "observations": observations,
                            "authority_scope": "reservation held only during trusted proposal subprocess"},
        "child_process": process, "artifacts": artifacts, "output": str(output),
        "elapsed_ms": max(0, int((time.monotonic() - started) * 1000)), **_FALSE})
    record["result_cid"] = cid_for_structured(record)
    result_raw = canonical_dag_json_bytes(record)
    _write(output / "result.json", result_raw)
    expected_files[str(output / "result.json")] = result_raw
    verify_generated_finite_repository_candidate(record=record)
    # All native, signature and historical verifier callbacks precede this
    # detached source/CAS/AST/executable/producer/artifact closure.
    fence()
    _need(lease.released is True, "candidate reservation remains held")
    checkpoint()
    return record


def verify_generated_finite_repository_candidate(*, record):
    """Pure historical reconstruction; never fresh authority or child attestation."""
    value = _plain(record)
    _need(type(value) is dict and set(value) == _RESULT_FIELDS and value["schema"] == RESULT_SCHEMA
          and value["profile"] == PROFILE and value["status"] == "candidate_generated"
          and value["result_cid"] == cid_for_structured({key: row for key, row in value.items() if key != "result_cid"})
          and all(value[name] is False for name in _FALSE)
          and all(value[name] is True for name in ("canonical_source_unchanged", "proposal_generated",
              "observed_current_on_return", "reservation_released_on_return"))
          and type(value["provider_calls"]) is int and value["provider_calls"] == 0
          and type(value["training_steps"]) is int and value["training_steps"] == 0
          and type(value["elapsed_ms"]) is int and value["elapsed_ms"] >= 0,
          "closed exact generated candidate record required")
    checked = verify_finite_repository_candidate(admission=value["parent_admission"], candidate=value["reviewed_candidate"])
    p = checked["payload"]
    _need(value["parent_admission_cid"] == p["parent_admission_cid"]
          and value["reviewed_candidate_cid"] == checked["candidate_cid"]
          and _same(value["head"], p["head"]) and value["task_cid"] == p["task_cid"]
          and value["source_cid"] == p["source_cid"] and value["replacement_cid"] == p["after_cid"],
          "generated candidate retargeted its signed parent or reviewed task")
    output = Path(value["output"])
    _need(output.is_absolute() and output.resolve(strict=True) == output and not output.is_symlink()
          and output.is_dir() and set(value["artifacts"]) == set(_ARTIFACT_NAMES),
          "complete owned generated artifact inventory required")
    raws = {}
    for role, name in _ARTIFACT_NAMES.items():
        row = value["artifacts"][role]
        _need(type(row) is dict and set(row) == {"path", "bytes", "sha256", "cid"}
              and row["path"] == str(output / name) and type(row["bytes"]) is int,
              "exact generated artifact descriptor required")
        raw = _read(output / name, finite.MAX_BYTES)
        _need(row == _pin(output / name, raw), "generated artifact bytes changed")
        raws[role] = raw
    _need(raws["candidate"] == canonical_dag_json_bytes(value["reviewed_candidate"])
          and raws["source"] == BEFORE_SOURCE and raws["replacement"] == AFTER_SOURCE
          and raws["driver"] == _DRIVER and raws["process"] == canonical_dag_json_bytes(value["child_process"]),
          "generated artifact cannot be rebuilt from reviewed parent")
    receipt = json.loads(raws["child_receipt"])
    _need(type(receipt) is dict and set(receipt) == {"schema", "candidate_sha256", "task_cid", "before_sha256",
          "after_sha256", "pid", "uid", "provider_calls", "publication_authority", "completion_authority", "environment_keys"}
          and receipt["schema"] == "finite-repository-trusted-child-receipt@1"
          and receipt["candidate_sha256"] == _sha(raws["candidate"]) and receipt["task_cid"] == p["task_cid"]
          and receipt["before_sha256"] == _sha(BEFORE_SOURCE) and receipt["after_sha256"] == _sha(AFTER_SOURCE)
          and type(receipt["pid"]) is int and receipt["pid"] > 0 and type(receipt["uid"]) is int and receipt["uid"] >= 0
          and type(receipt["provider_calls"]) is int and receipt["provider_calls"] == 0
          and receipt["publication_authority"] is False and receipt["completion_authority"] is False
          and receipt["environment_keys"] == sorted([*_ENV, "HOME", "TMPDIR", "TMP", "TEMP"]),
          "trusted child receipt differs from the closed proposal contract")
    process = value["child_process"]
    _need(type(process) is dict and set(process) == _PROCESS_FIELDS
          and process["interface_version"] == native_process.BOUNDED_TOOL_RUNNER_VERSION
          and process["runtime"] == "native"
          and process["command"] == [p["python"]["path"], "-I", "-S", "driver.py", _sha(raws["candidate"])]
          and type(process["returncode"]) is int and process["returncode"] == 0
          and process["stdout"] == process["stderr"] == "" and process["workspace_cleaned"] is True
          and _same(process["limits"], p["limits"]) and type(process["pid"]) is int and process["pid"] == receipt["pid"]
          and type(process["elapsed_ms"]) is int and 0 <= process["elapsed_ms"] <= 5000
          and type(process["effective_timeout_ms"]) is int and 1 <= process["effective_timeout_ms"] <= 5000
          and process["termination_reason"] == "completed" and process["error"] == ""
          and all(process[name] is False for name in ("timed_out", "cancelled", "unavailable", "output_truncated",
              "workspace_limit_exceeded", "process_tree_terminated", "resource_exhausted")),
          "successful exact native child record required")
    _verify_capacity(value["native_capacity"], process["pid"])
    _need(raws["source_custody"] == canonical_dag_json_bytes(value["source_custody"]),
          "immutable source custody artifact differs")
    _verify_custody(value["source_custody"], value["parent_admission"])
    current = json.loads(raws["current_verification"])
    _need(type(current) is dict and set(current) == {"schema", "admission_cid", "fresh_evidence", "semantic_context", "observed_current", *finite._FALSE}
          and current["schema"] == "finite-repository-current-verification@1"
          and current["admission_cid"] == value["parent_admission_cid"] and current["observed_current"] is True
          and all(current[name] is False for name in finite._FALSE)
          and _same(current["semantic_context"], finite._evidence(current["fresh_evidence"],
              value["parent_admission"]["declaration"], value["parent_admission"]["graph"]))
          and _same(current["semantic_context"], finite.verify_finite_repository_admission(
              admission=value["parent_admission"])["semantic_context"]), "retained fresh preparation bindings differ")
    _need(_read(output / "result.json", finite.MAX_BYTES) == canonical_dag_json_bytes(value),
          "immutable generated result bytes changed")
    return {"schema": "finite-repository-generated-candidate-historical-verification@1",
        "result_cid": value["result_cid"], "observed_current": False, "child_origin_authenticated": False, **_FALSE}


__all__ = ["PROFILE", "CANDIDATE_SCHEMA", "RESULT_SCHEMA", "BEFORE_SOURCE", "AFTER_SOURCE",
    "FiniteRepositoryCandidateError", "author_finite_repository_candidate", "verify_finite_repository_candidate",
    "generate_finite_repository_candidate", "verify_generated_finite_repository_candidate"]
