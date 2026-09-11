"""Launcher-owned independent source verification on the native kit store.

No RPC accepts evidence, issuer credentials, executable paths or callbacks.
Read admission only observes persisted records and current native bindings.
This initial producer never issues SPAR accepted-root/completion authority.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from .kit_source_forest import source_manifest, OWNER_FIELDS
from ..task_sources.control_plane_contracts import content_identity

REQUEST_SCHEMA = "accelerator/spar-native-verification-request@1"
EXECUTION_SCHEMA = "accelerator/spar-native-verification-execution@1"
MAX_ARCHIVE_BYTES = 32 * 1024**2
SEALS = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
WORKER = "ipfs_accelerate_py/agent_supervisor/semantic_state/spar_verifier_worker.py"
REQUIRED_MISSING = (
    "all_required_language_and_dynamic_semantics", "required_mode_roots_and_transitions",
    "differential_trace_selection_and_proof_execution",
    "noncompensable_safety_floors_with_failure_denominators",
    "self_hosted_capstone_and_procedure_reuse", "two_epoch_seven_slice_fixed_point",
    "native_runtime_callback_and_merge_settlement",
)


class NativeVerificationUnavailable(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _git(root, *args):
    return subprocess.run(["/usr/bin/git", "-c", "core.fsmonitor=false", *args],
        cwd=root, check=True, capture_output=True, timeout=10,
        env={"PATH": "/usr/bin:/bin", "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1",
             "GIT_NO_REPLACE_OBJECTS": "1",
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}).stdout


def _process(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "parent_pid": int(fields[1]), "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _sealed_fd(name, raw=None):
    fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        if raw is not None:
            with os.fdopen(os.dup(fd), "wb") as output:
                output.write(raw)
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
        return fd
    except BaseException:
        os.close(fd)
        raise


def capture_verifier_archive(repository_root, source):
    """Capture exact committed native worker and datasets contract modules."""
    root = Path(repository_root).resolve(strict=True)
    forest = source["source_forest"]
    specs = [r for r in forest["nested_repositories"] if r["repository"] == "ipfs_datasets"]
    if len(specs) != 1:
        raise NativeVerificationUnavailable("one_selected_datasets_repository_required")
    spec = specs[0]
    datasets = (root / spec["path"]).resolve(strict=True)
    if not datasets.is_relative_to(root):
        raise NativeVerificationUnavailable("datasets_source_escaped_forest")
    worker = _git(root, "cat-file", "blob", forest["source_head"] + ":" + WORKER)
    if (root / WORKER).is_symlink() or (root / WORKER).read_bytes() != worker:
        raise NativeVerificationUnavailable("native_worker_source_changed")
    paths = _git(datasets, "ls-tree", "-r", "--name-only", "-z", spec["head"],
                 "--", "ipfs_datasets_py/logic/software_contracts/").split(b"\0")
    captured = {"__main__.py": worker}
    size = len(worker)
    dependencies = []
    # Capture the small installed CID dependency closure into the same sealed
    # archive. The child has no site-packages lookup. These exact observed
    # dependency bytes/versions are in its owner request, not a claimed Git pin.
    for distribution, prefix in (("multiformats", "multiformats/"),
            ("multiformats-config", "multiformats_config/"), ("bases", "bases/"),
            ("typing-validation", "typing_validation/"), ("typing-extensions", "typing_extensions.py")):
        package = importlib.metadata.distribution(distribution)
        files = {}
        for entry in package.files or ():
            name = str(entry)
            if not (name.startswith(prefix) and (name.endswith(".py") or name.endswith(".json"))):
                continue
            path = Path(package.locate_file(entry))
            if path.is_symlink() or not path.is_file():
                raise NativeVerificationUnavailable("CID_dependency_source_unavailable")
            data = path.read_bytes()
            size += len(data)
            if size > MAX_ARCHIVE_BYTES:
                raise NativeVerificationUnavailable("CID_dependency_source_exceeded_bound")
            files[name] = hashlib.sha256(data).hexdigest()
            captured[name] = data
        if not files:
            raise NativeVerificationUnavailable("CID_dependency_inventory_unavailable")
        dependencies.append({"distribution": distribution, "version": package.version, "files": files})
    for raw in paths:
        if not raw or not raw.endswith(b".py"):
            continue
        relative = raw.decode("utf-8", "strict")
        path = datasets / relative
        data = _git(datasets, "cat-file", "blob", spec["head"] + ":" + relative)
        size += len(data)
        if size > MAX_ARCHIVE_BYTES or path.is_symlink() or path.read_bytes() != data:
            raise NativeVerificationUnavailable("datasets_verifier_source_changed_or_exceeded_bound")
        captured[relative] = data
    if "ipfs_datasets_py/logic/software_contracts/semantic_state/spar_verification.py" not in captured:
        raise NativeVerificationUnavailable("datasets_native_verifier_unavailable")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, data in sorted(captured.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            output.writestr(info, data)
    data = archive.getvalue()
    return data, {"archive_sha256": hashlib.sha256(data).hexdigest(),
                  "datasets_head": spec["head"], "worker_head": forest["source_head"],
                  "module_count": len(captured), "uncompressed_bytes": size,
                  "observed_CID_dependencies": dependencies}


def execute_verifier(request, archive):
    """Launch only the sealed native worker; retain actual child custody."""
    fds, child, pidfd = [], None, None
    started = time.monotonic_ns()
    try:
        archive_fd = _sealed_fd("spar-reviewed-verifier", archive); fds.append(archive_fd)
        request_fd = _sealed_fd("spar-owner-request", _json(request)); fds.append(request_fd)
        result_fd = _sealed_fd("spar-child-result"); fds.append(result_fd)
        executable = Path(sys.executable).resolve(strict=True)
        interpreter_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
        child = subprocess.Popen([str(executable), "-I", "-S", "-B",
            f"/proc/self/fd/{archive_fd}", str(archive_fd), str(request_fd), str(result_fd)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=tuple(fds), cwd="/", env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8",
                "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "core.fsmonitor", "GIT_CONFIG_VALUE_0": "false",
                "GIT_CONFIG_KEY_1": "core.hooksPath", "GIT_CONFIG_VALUE_1": "/dev/null"})
        pidfd = os.pidfd_open(child.pid)
        identity = _process(child.pid)
        if identity["parent_pid"] != os.getpid():
            raise NativeVerificationUnavailable("verifier_parent_changed")
        if not select.select([child.stdout], [], [], 20)[0]:
            raise NativeVerificationUnavailable("verifier_handshake_unavailable")
        handshake = child.stdout.readline(4096)
        if (not handshake.endswith(b"\n") or json.loads(handshake) != identity
                or _process(child.pid) != identity
                or hashlib.sha256(Path(f"/proc/{child.pid}/exe").read_bytes()).hexdigest() != interpreter_sha):
            raise NativeVerificationUnavailable("verifier_process_binding_changed")
        stdout, stderr = child.communicate(b"run\n", timeout=125)
        if child.returncode != 0 or stdout or stderr:
            raise NativeVerificationUnavailable("verifier_execution_failed:" + stderr.decode(errors="replace")[-512:])
        if not select.select([pidfd], [], [], 0)[0]:
            raise NativeVerificationUnavailable("verifier_exit_not_observed")
        size = os.fstat(result_fd).st_size
        if not 1 <= size <= 64 * 1024**2 or fcntl.fcntl(result_fd, fcntl.F_GET_SEALS) != SEALS:
            raise NativeVerificationUnavailable("verifier_result_not_sealed_or_bounded")
        result = json.loads(os.pread(result_fd, size, 0))
        if result.get("process_identity") != identity or result.get("accepted_root") is not False:
            raise NativeVerificationUnavailable("verifier_result_identity_changed")
        return result, {"process_identity": identity, "interpreter_sha256": interpreter_sha,
                        "started_monotonic_ns": started, "finished_monotonic_ns": time.monotonic_ns(),
                        "returncode": 0, "pidfd_exit_observed": True, "sealed_result": True}
    except subprocess.TimeoutExpired as exc:
        raise NativeVerificationUnavailable("native_verifier_deadline_exceeded") from exc
    finally:
        if child is not None and child.poll() is None:
            # This is only our newly launched, read-only verifier child. Never
            # signal a task actor, numeric PID, process group or descendant tree.
            if pidfd is not None:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                child.wait(timeout=10)
            else:
                # No run handshake was sent without retained custody. EOF
                # makes the fixed child refuse naturally, without PID signals.
                if child.stdin is not None:
                    child.stdin.close()
                child.wait(timeout=25)
        if pidfd is not None:
            os.close(pidfd)
        if child is not None:
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
        for fd in fds:
            os.close(fd)


def board_binding(facts, completion_projection):
    """Bind actual task/goal/claim/receipt rows; no expiry or settlement guess."""
    return {"facts": copy.deepcopy(dict(facts)), "completion_projection": copy.deepcopy(dict(completion_projection))}


class NativeSparVerification:
    """Private launcher producer with a strictly read-only observer."""
    def __init__(self, persistence, repository_root, *, connection, transaction_lock, owner_identity):
        self.persistence = persistence
        self.root = str(Path(repository_root).resolve())
        self.namespace = persistence.namespace + "/native-source-verification"
        # Kit deliberately forbids retargeting a bound store. Bind a second
        # closed namespace to the SAME admitted connection, lock and owner;
        # this opens no database or writer and cannot alter the source root.
        self.store = type(persistence.store)(connection, transaction_lock=transaction_lock,
            owner_identity=owner_identity, namespace=self.namespace)
        self._running = threading.Lock()
        # A content record cannot invent native execution custody. Only this
        # living producer's completed, pidfd-observed executions enter here.
        # Owner replacement deliberately requires a fresh execution.
        self._issued_records = {}

    def _capture(self, connection, task_cids):
        from ..task_sources.closeout_snapshot import capture_closeout_facts
        from ..task_sources.intent_repository import completion_evidence_projection_on_connection
        from ..task_sources.spar_closeout_profile import observe_source
        self.persistence.store.current_state_root(self.persistence.namespace)
        connection.execute("BEGIN TRANSACTION")
        try:
            binding = board_binding(capture_closeout_facts(connection),
                completion_evidence_projection_on_connection(connection, task_cids=task_cids,
                                                              transaction_owned_by_caller=True))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        source = observe_source(self.root, self.persistence.profile["nested_repositories"])
        return source, binding

    def run(self, connection, transaction_lock, task_cids):
        if not self._running.acquire(blocking=False):
            raise NativeVerificationUnavailable("native_verification_already_running")
        try:
            with transaction_lock:
                source, binding = self._capture(connection, task_cids)
                request = self.make_request(source, binding)
                current = self.observe(source, binding)
                if current.get("admitted") is True:
                    return {**current, "idempotent_replay": True}
            archive, code_identity = capture_verifier_archive(self.root, source)
            # Bind the exact sealed code and interpreter to the immutable
            # owner request before any execution is admitted.
            request["verifier_source"] = code_identity
            request["interpreter_sha256"] = hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
            request["native_execution_nonce"] = uuid.uuid4().hex
            with transaction_lock:
                fresh_source, fresh_binding = self._capture(connection, task_cids)
                if self.make_request(fresh_source, fresh_binding) != {k:v for k,v in request.items()
                        if k not in {"verifier_source", "interpreter_sha256", "native_execution_nonce"}}:
                    raise NativeVerificationUnavailable("native_inputs_changed_before_execution")
                request_cid = self.store.put(request, codec="dag-json", replicate=False)["cid"]
            child, execution = execute_verifier(request, archive)
            if (child.get("request_cid") != request_cid or child.get("completion_authority") is not False
                    or execution["interpreter_sha256"] != request["interpreter_sha256"]):
                raise NativeVerificationUnavailable("native_child_request_or_interpreter_changed")
            with transaction_lock:
                fresh_source, fresh_binding = self._capture(connection, task_cids)
                if self.make_request(fresh_source, fresh_binding) != {k:v for k,v in request.items()
                        if k not in {"verifier_source", "interpreter_sha256", "native_execution_nonce"}}:
                    raise NativeVerificationUnavailable("native_inputs_changed_after_execution")
                store = self.store
                outcome = child["outcome"]
                block_cids = []
                if outcome["status"] == "verified_component":
                    for cid, block in outcome.pop("blocks").items():
                        store.put(block, expected_cid=cid, codec="dag-json", replicate=False)
                        block_cids.append(cid)
                    store.put(outcome.pop("report"), expected_cid=outcome["report_cid"], codec="dag-json", replicate=False)
                result_cid = store.put(child, codec="dag-json", replicate=False)["cid"]
                # Immutable block writes may take time. Recheck the full
                # source/board binding before publishing their execution root.
                fresh_source, fresh_binding = self._capture(connection, task_cids)
                if self.make_request(fresh_source, fresh_binding) != {k:v for k,v in request.items()
                        if k not in {"verifier_source", "interpreter_sha256", "native_execution_nonce"}}:
                    raise NativeVerificationUnavailable("native_inputs_changed_during_persistence")
                record = {"schema": EXECUTION_SCHEMA, "request_cid": request_cid, "result_cid": result_cid,
                          "owner_identity": self.persistence.owner_identity, "execution": execution,
                          "semantic_block_cids": sorted(block_cids),
                          "accepted_root": False, "completion_authority": False}
                record_cid = store.put(record, codec="dag-json", replicate=False)["cid"]
                before = store.current_state_root(self.namespace)
                published = store.compare_and_swap_state_root(self.namespace,
                    expected_revision=before["revision"], expected_root_cid=before["root_cid"],
                    new_root_cid=record_cid, operation_id="native-verifier:" + request_cid)
                if published["status"] not in {"updated", "unchanged"}:
                    raise NativeVerificationUnavailable("native_verification_publication_conflict")
                self._issued_records[record_cid] = copy.deepcopy(record)
                return self.observe(fresh_source, fresh_binding)
        finally:
            self._running.release()

    def make_request(self, source, binding):
        persistence = self.persistence
        kit = persistence.observe(source)
        if kit.get("admitted") is not True:
            raise NativeVerificationUnavailable("current_kit_source_forest_required")
        forest = source["source_forest"]
        repositories = [{"repository": "ipfs_accelerate", "path": self.root,
                         "head": forest["source_head"], "tree": source["repository_tree_id"]}]
        repositories.extend({"repository": r["repository"], "path": str(Path(self.root) / r["path"]),
                             "head": r["head"], "tree": r["tree"]} for r in forest["nested_repositories"])
        return {"schema": REQUEST_SCHEMA, "profile_cid": persistence.profile_cid,
                "profile": copy.deepcopy(persistence.profile), "owner_identity": persistence.owner_identity,
                "source_manifest": source_manifest(persistence.profile, persistence.profile_cid, source),
                "kit_source_forest": kit, "board_binding": binding, "repositories": repositories,
                "accepted_root": False, "completion_authority": False}

    def observe(self, source, binding):
        result = {"admitted": False, "accepted_root": False, "completion_authority": False,
                  "semantic_acceptance_authority": False, "missing_coverage": list(REQUIRED_MISSING)}
        try:
            store = self.store
            current = store.current_state_root(self.namespace)
            if not current.get("root_cid"):
                raise NativeVerificationUnavailable("native_verifier_execution_required")
            record = store.get(current["root_cid"])
            request = self.make_request(source, binding)
            stored_request = store.get(record["request_cid"])
            if (record != self._issued_records.get(current["root_cid"])
                    or record.get("schema") != EXECUTION_SCHEMA
                    or {k:v for k,v in stored_request.items() if k not in {"verifier_source", "interpreter_sha256", "native_execution_nonce"}} != request):
                raise NativeVerificationUnavailable("verification_current_binding_changed")
            transition = store.get(current["transition_cid"])
            if (transition["new_root_cid"] != current["root_cid"]
                    or transition["new_revision"] != current["revision"]
                    or transition["namespace"] != self.namespace
                    or record.get("completion_authority") is not False
                    or record.get("accepted_root") is not False):
                raise NativeVerificationUnavailable("verification_record_or_transition_invalid")
            execution = record["execution"]
            if (execution["returncode"] != 0 or execution["pidfd_exit_observed"] is not True
                    or execution["sealed_result"] is not True
                    or record["owner_identity"] != self.persistence.owner_identity):
                raise NativeVerificationUnavailable("native_execution_custody_unverified")
            child = store.get(record["result_cid"])
            if child["request_cid"] != record["request_cid"] or child["process_identity"] != execution["process_identity"]:
                raise NativeVerificationUnavailable("child_request_binding_changed")
            outcome = child["outcome"]
            if outcome["status"] != "verified_component":
                return {**result, "reason": "native_verifier_unavailable", "outcome": outcome}
            report = store.get(outcome["report_cid"])
            if (report["schema"] != "ipfs-datasets/spar-source-verification@1"
                    or report["missing_coverage"] != list(REQUIRED_MISSING)
                    or report["accepted_root"] is not False or report["completion_authority"] is not False
                    or report["semantic_acceptance_authority"] is not False):
                raise NativeVerificationUnavailable("semantic_report_contract_changed")
            for cid in record["semantic_block_cids"]:
                store.get(cid)
            from ..task_sources.spar_closeout_profile import observe_source
            after = observe_source(self.root, self.persistence.profile["nested_repositories"])
            if source_manifest(self.persistence.profile, self.persistence.profile_cid, after) != request["source_manifest"]:
                raise NativeVerificationUnavailable("source_changed_during_read_admission")
            return {**result, "admitted": True, "authority": "native_executed_source_component_only",
                    "request_cid": record["request_cid"], "execution_cid": current["root_cid"],
                    "root_revision": current["revision"], "report": report}
        except Exception as exc:
            return {**result, "reason": "native_verification_not_current", "detail": str(exc)[:512]}
