"""Experimental exact-key index of checked *conditional model* evidence.

This is a bounded experiment catalog, not the supervisor's proof authority or
task-admission cache. Production is through the existing native learned-decoder
and checker routes. An externally pinned manifest is required for every read;
SQL status text, old qualification markers, and caller eligibility flags confer
no authority. Full datasets and supervisor cache keys are retained together.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import terminal_codebase_decoder_logic as decoder_logic
from . import terminal_codebase_logic_qualification as logic

SCHEMA = "terminal-codebase-proof-index@1"
RELATION_SCHEMA = "terminal-codebase-proof-key-relationship@1"
LOOKUP_SCHEMA = "terminal-codebase-model-evidence-lookup@1"
MAX_DATABASE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ENV_FILES = 10000
MAX_ENV_BYTES = 3 * 1024 * 1024 * 1024
AUTHORITY = {**logic.AUTHORITY, "behavioral_satisfaction": False}
SMT_BOUNDS = {"timeout_ms": 5000, "max_steps": 100000,
    "max_memory_bytes": 128 * 1024 * 1024, "max_output_bytes": 65536}
LEAN_BOUNDS = {"timeout_seconds": 20, "cpu_seconds": 20,
    "max_input_bytes": 262144, "max_output_bytes": 65536,
    "max_workspace_bytes": 16 * 1024 * 1024}
POLICY = {"schema": "terminal-conditional-model-index-policy@1",
    "profile": "public_bottle_two_header_normalizers_v1", "entry_count": 7,
    "authority_ceiling": "bounded", "scope": "checked_conditional_model_only",
    "runtime_behavior": "unknown", "network": "disabled",
    "source_execution": False, "optimizer_steps": 0, **AUTHORITY}


def _wire(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("proof-index JSON budget exceeded")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(value):
    return "sha256:" + _sha(_wire(value))


def _file(path, maximum=MAX_ENV_BYTES):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("missing, indirect, or oversized bound artifact: " + str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1048576), b""):
            size += len(raw)
            if size > maximum:
                raise ValueError("artifact byte budget exceeded")
            digest.update(raw)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _read_json(path):
    pin = _file(path, MAX_JSON_BYTES)
    value = json.loads(Path(path).read_bytes())
    if _file(path, MAX_JSON_BYTES) != pin:
        raise ValueError("artifact changed during read")
    _wire(value)
    return value


def _inert(value):
    raw = _wire(value)
    return {"canonical_json": raw.decode(), "sha256": _sha(raw)}


def _source_context(source_bytes, source_path, decoder_experiment):
    from .terminal_codebase_decoder_training import validate_terminal_codebase_decoder
    from .terminal_codebase_decoder_experiment import _captured_source

    if type(source_bytes) is not bytes or not 0 < len(source_bytes) <= 1048576:
        raise ValueError("bounded exact source bytes required")
    if source_path != "bottle.py":
        raise ValueError("experimental profile requires the exact public bottle.py path")
    if decoder_experiment.get("schema") != "terminal-codebase-decoder-experiment@1":
        raise ValueError("native learned decoder experiment required")
    if _read_json(Path(decoder_experiment["output"]) / "result.json") != decoder_experiment:
        raise ValueError("frozen native decoder experiment differs from its complete artifact")
    original, capture = _captured_source(decoder_experiment["parent_capture"]["prepared_experiment"])
    if original != source_bytes or _wire(capture) != _wire(decoder_experiment["parent_capture"]):
        raise ValueError("current complete source/dependency capture differs")
    decoder = validate_terminal_codebase_decoder(expected=decoder_experiment["decoder"],
        source_bytes=source_bytes, source_path=source_path)
    snapshot = decoder_experiment["source_snapshot"]
    if (snapshot["public_source_sha256"] != _sha(source_bytes)
            or snapshot["decoder_checkpoint"] != decoder["checkpoint"]
            or snapshot["decoder_report_sha256"] != _sha(decoder_logic._wire(decoder))
            or snapshot["parent_ir_manifest_id"] != capture["parent_ir_manifest_id"]
            or snapshot["parent_preplanning_manifest_sha256"] != capture["parent_manifest"]["sha256"]
            or snapshot["source_roles"] != decoder["policy"]["role_policy"]):
        raise ValueError("decoder source snapshot differs from independently replayed inputs")
    source = {"source_path": source_path, "source_sha256": _sha(source_bytes),
        "source_bytes": len(source_bytes), "parent_capture": capture,
        "decoder_source_snapshot": snapshot, "checkpoint": decoder["checkpoint"],
        "decoder_policy": _inert(decoder["policy"]),
        "decoder_report_sha256": _sha(decoder_logic._wire(decoder)),
        "candidate_inference": [{"unit_id": row["unit_id"],
            "inference_sha256": row["inference_sha256"],
            "candidate_source_sha256": None if row["candidate_source"] is None else _sha(row["candidate_source"].encode()),
            "source_binding": row["source_binding"], "source_ast_sha256": row["source_ast_sha256"]}
            for row in decoder["candidates"]]}
    return source, decoder


def _translator():
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    from ipfs_datasets_py.logic.backends.smt.compiler import SoftwareVerificationSMTCompiler
    from ipfs_datasets_py.logic.backends.z3.compiler import Z3SoftwareVerificationBackend
    from ipfs_datasets_py.logic.backends.process import BoundedToolRunner
    from ipfs_datasets_py.logic.common import canonical_cache_key
    from ipfs_accelerate_py.agent_supervisor.proof import formal_verification_cache, proof_scope_index
    from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as code_header_guard
    modules = [header, canonical_cache_key, formal_verification_cache, proof_scope_index,
        decoder_logic, logic, code_header_guard]
    paths = {Path(module.__file__).resolve() for module in modules}
    paths.update(Path(inspect.getfile(cls)).resolve() for cls in (
        SoftwareVerificationSMTCompiler, Z3SoftwareVerificationBackend, BoundedToolRunner))
    # Bind all source modules in the native logic/backend/compiler and learned
    # decoder packages, including transitive helpers, not only exported classes.
    datasets = Path(header.__file__).resolve().parents[1]
    for relative in ("backends", "security_ir", "ir_core", "formalization/autoencoder/security"):
        paths.update((datasets / relative).rglob("*.py"))
    from . import terminal_codebase_decoder_training as training
    paths.add(Path(training.__file__).resolve())
    paths.add(Path(__file__).resolve())
    if len(paths) > 2048:
        raise ValueError("translator inventory budget exceeded")
    return {"schema": "terminal-codebase-translator-snapshot@1",
        "files": [_file(path, 4 * 1024 * 1024) for path in sorted(paths)],
        "header_profile": header.describe_header_semantics_profile()}


def _version(executable, *args):
    completed = subprocess.run([str(executable), *args], capture_output=True,
        timeout=5, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    if completed.returncode != 0 or len(completed.stdout) + len(completed.stderr) > 65536:
        raise ValueError("native checker identity probe failed")
    return completed.stdout.decode().strip()


def capture_terminal_codebase_proof_environment(*, lean_executable=None, z3_executable="z3"):
    """Pin the installed checker packages and their actual shared dependencies."""
    import duckdb
    lean, unavailable = logic._native_lean(lean_executable)
    z3 = shutil.which(z3_executable)
    if unavailable or lean is None or z3 is None or sys.platform != "linux":
        raise ValueError("installed native Linux Lean/Z3 profile required")
    lean, z3 = lean.resolve(), Path(z3).resolve()
    root = lean.parent.parent
    files = {lean, z3, Path(sys.executable).resolve()}
    shared = set()
    for path in (root / "lib").rglob("*"):
        if path.is_file() and (".olean" in path.name or ".so" in path.name):
            files.add(path.resolve())
            if ".so" in path.name:
                with path.open("rb") as stream:
                    if stream.read(4) == b"\x7fELF":
                        shared.add(path.resolve())
    for executable in sorted(shared | {lean, z3, Path(sys.executable).resolve()}):
        text = _version("/usr/bin/ldd", str(executable))
        for line in text.splitlines():
            for part in line.split():
                if part.startswith("/") and Path(part).is_file():
                    files.add(Path(part).resolve())
    if len(files) > MAX_ENV_FILES:
        raise ValueError("checker environment inventory budget exceeded")
    pins, total = [], 0
    for path in sorted(files):
        pin = _file(path)
        total += pin["bytes"]
        if total > MAX_ENV_BYTES:
            raise ValueError("checker environment byte budget exceeded")
        pins.append(pin)
    # DuckDB affects lookup semantics; torch/numpy affect the independently
    # replayed checkpoint inference. Their versions and native module bytes
    # join the checker environment identity.
    package_pins = []
    for name in ("duckdb", "torch", "numpy"):
        module = __import__(name)
        path = Path(module.__file__).resolve()
        package_pins.append({"name": name, "version": importlib.metadata.version(name),
            "module": _file(path, 16 * 1024 * 1024)})
    return {"schema": "terminal-codebase-native-proof-environment@1",
        "system": platform.system(), "machine": platform.machine(),
        "python": sys.version, "files": pins, "total_file_bytes": total,
        "packages": package_pins, "duckdb_version": duckdb.__version__,
        "lean": {"executable": str(lean), "version": _version(lean, "--version"),
            "sha256": _file(lean)["sha256"]},
        "z3": {"executable": str(z3), "version": _version(z3, "--version"),
            "sha256": _file(z3)["sha256"]},
        "execution_environment": {"network": "disabled", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}}


def build_terminal_codebase_proof_key_relationship(*, dimensions):
    """Losslessly relate two native key bodies to their complete raw inputs."""
    from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import ProofCacheKey
    required = {"source", "expression", "formalization", "slice", "obligation", "assumptions",
        "bounds", "translation", "provider", "environment", "policy", "schema", "checker",
        "network_policy", "evidence_kind", "authority_ceiling", "kernel", "theorem_registry"}
    if type(dimensions) is not dict or set(dimensions) != required:
        raise ValueError("closed complete proof-key dimensions required")
    canonical = CanonicalProofCacheKey.build(**{key: dimensions[key] for key in required
        if key not in {"kernel", "theorem_registry"}})
    supervisor = ProofCacheKey(obligation=dimensions["obligation"],
        premises=tuple(dimensions["assumptions"]), translator=dimensions["translation"],
        solver={"provider": dimensions["provider"], "checker": dimensions["checker"],
            "environment": dimensions["environment"]}, kernel=dimensions["kernel"],
        toolchain=dimensions["environment"], theorem_registry=dimensions["theorem_registry"],
        policy=dimensions["policy"], resource_budget=dimensions["bounds"],
        candidate_tree={"source": dimensions["source"], "slice": dimensions["slice"],
            "expression": dimensions["expression"], "formalization": dimensions["formalization"],
            "schema": dimensions["schema"], "network_policy": dimensions["network_policy"],
            "evidence_kind": dimensions["evidence_kind"], "authority_ceiling": dimensions["authority_ceiling"]})
    result = {"schema": RELATION_SCHEMA, "dimensions": dimensions,
        "datasets_key": canonical.to_dict(), "datasets_key_id": canonical.key_id,
        "accelerate_key": supervisor.to_dict(), "accelerate_key_id": supervisor.key_id}
    result["relationship_id"] = _identity(result)
    return result


def _qualification_semantics(value):
    # Paths/timings identify a local invocation. All semantic fields, native
    # scripts/models, receipt outcomes, compiled object hashes, and frontiers
    # must remain equal; no arbitrary claimed status survives comparison.
    if type(value) is list:
        return [_qualification_semantics(item) for item in value]
    if type(value) is dict:
        return {key: _qualification_semantics(item) for key, item in value.items()
            if key not in {"output", "path", "elapsed_seconds", "qualification_sha256"}}
    return value


def _artifact_receipts(qualification):
    for receipt in qualification["lean_checks"]:
        for pin in [receipt["artifact"], *receipt["compiled_artifacts"]]:
            if _file(pin["path"], 16 * 1024 * 1024) != pin:
                raise ValueError("bound Lean source/object artifact changed")


def _environment_reference(environment):
    return {"schema": "terminal-codebase-proof-environment-ref@1",
        "environment_sha256": _sha(_wire(environment)), "lean": environment["lean"],
        "z3": environment["z3"], "python": environment["python"], "packages": environment["packages"]}


def _entries(source, qualification, environment, translation):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    from ipfs_datasets_py.logic.backends.smt.compiler import SoftwareVerificationSMTCompiler, SmtObligation

    derivation = header.validate_header_semantics(qualification["native_header_derivation"],
        source_bytes=Path(source["parent_capture"]["source"]["path"]).read_bytes(),
        source_path=source["source_path"], protocol=header.WsgiHeaderProtocolContract(logic.REVIEW_PREMISE))
    checks = qualification["native_smt_check"]
    if (qualification["status"] != "qualified_learned_header_model"
            or qualification["candidate_generation"] != "learned_decoder"
            or checks["status"] != "checked_local_model" or checks["solver_calls"] != 6
            or checks["solver_executable_sha256"] != environment["z3"]["sha256"]):
        raise ValueError("independently replayed complete native model checking required")
    premises = [*derivation["assumptions"], {"reviewed_protocol": derivation["contract"]["protocol"]}]
    common = {"source": source, "assumptions": premises, "translation": translation,
        "environment": _environment_reference(environment), "policy": POLICY, "schema": {"index": SCHEMA,
            "relation": RELATION_SCHEMA, "derivation": derivation["schema"]},
        "network_policy": {"network": "disabled", "provider_calls": 0},
        "authority_ceiling": "bounded"}
    rows = []
    if len(checks["results"]) != 6 or len(derivation["smt_targets"]) != 6:
        raise ValueError("exact six native model checks required")
    for target, receipt in zip(derivation["smt_targets"], checks["results"], strict=True):
        compiled = SoftwareVerificationSMTCompiler().compile(SmtObligation.from_dict(target["obligation"]))
        if (compiled.to_dict() != target["compilation"]
                or receipt["symbol"] != target["symbol"] or receipt["kind"] != target["kind"]
                or receipt["compilation_id"] != compiled.compilation_id
                or receipt["script_sha256"] != _sha(compiled.smtlib.encode())
                or receipt["script_digest"] != compiled.script.digest
                or receipt["solver_answer"] != target["expected_model_answer"]
                or receipt["query_mode"] != target["obligation"]["query_mode"]
                or receipt["solver_version"] != environment["z3"]["version"]
                or receipt["matches_model_expectation"] is not True):
            raise ValueError("exact native SMT compilation/check relation differs")
        units = [item for item in derivation["modeled_symbols"] if item["symbol"] == target["symbol"]]
        dims = {**common, "slice": units, "expression": compiled.smtlib,
            "formalization": target["compilation"], "obligation": target["obligation"],
            "bounds": SMT_BOUNDS, "provider": "native-z3-header-model-v1",
            "checker": "native-z3-software-verification-v1", "evidence_kind": "solver_result",
            "kernel": {"scope": "SMT_solver_result_only_no_kernel_certificate"},
            "theorem_registry": {"derivation_cid": derivation["derivation_cid"],
                "check_cid": checks["check_cid"], "symbol": target["symbol"], "kind": target["kind"]}}
        rows.append(_entry(dims, symbol=target["symbol"], property=target["kind"],
            domain="conditional_header_string_model", classification="conditional_model_sat_witness"
            if receipt["solver_answer"] == "sat" else "conditional_model_unsat",
            units=units, premises=premises, receipt=receipt, derivation=derivation))
    positive, negative = qualification["lean_checks"]
    lean_source, theorem_ids = logic._header_lean(derivation)
    if (positive["status"] != "passed" or positive["returncode"] != 0
            or positive["expected_success"] is not True or positive["backend_executed"] is not True
            or positive["matches_expectation"] is not True or len(positive["compiled_artifacts"]) != 1
            or positive["artifact"]["sha256"] != _sha(lean_source.encode())
            or positive["executable_sha256"] != environment["lean"]["sha256"]
            or any(positive[key] for key in ("timed_out", "output_truncated", "workspace_limit_exceeded", "resource_exhausted"))
            or negative["status"] != "rejected" or negative["matches_expectation"] is not True):
        raise ValueError("actual positive Lean object and rejected guard control required")
    dims = {**common, "slice": derivation["modeled_symbols"], "expression": lean_source,
        "formalization": {"file": positive["file"], "source_sha256": positive["artifact"]["sha256"],
            "compiled_artifacts": positive["compiled_artifacts"], "theorem_ids": theorem_ids},
        "obligation": {"theorem_ids": theorem_ids, "scope": positive["proof_scope"]},
        "bounds": LEAN_BOUNDS, "provider": "native-lean-header-model-v1", "checker": "native-lean-kernel-v1",
        "evidence_kind": "kernel_checked_proof", "kernel": environment["lean"],
        "theorem_registry": {"theorem_ids": theorem_ids, "source": positive["artifact"]}}
    rows.append(_entry(dims, symbol="_hkey+_hval", property="header_boolean_model",
        domain="conditional_header_boolean_model", classification="conditional_model_kernel_checked",
        units=derivation["modeled_symbols"], premises=premises, receipt=positive, derivation=derivation))
    if len(rows) != 7 or len({item["entry_id"] for item in rows}) != 7:
        raise ValueError("exact unique seven-entry profile required")
    return rows


def _entry(dimensions, *, symbol, property, domain, classification, units, premises, receipt, derivation):
    relationship = build_terminal_codebase_proof_key_relationship(dimensions=dimensions)
    evidence = {"schema": "terminal-codebase-conditional-model-evidence@1", "symbol": symbol,
        "property": property, "model_domain": domain, "classification": classification,
        "source_path": dimensions["source"]["source_path"], "source_sha256": dimensions["source"]["source_sha256"],
        "source_unit_bindings": units, "premises": premises,
        "open_frontiers": derivation["open_frontiers"], "checker_receipt": receipt, **AUTHORITY}
    result = {"schema": "terminal-codebase-proof-index-entry@1", "key_relationship": relationship,
        "evidence": evidence}
    result["entry_id"] = _identity(result)
    return result


def _source_snapshot(source, qualification, environment, translation):
    return {"schema": "terminal-codebase-proof-source-snapshot@1",
        "source_path": source["source_path"], "source_sha256": source["source_sha256"],
        "source_bytes": source["source_bytes"],
        "source_unit_bindings": qualification["native_header_derivation"]["modeled_symbols"],
        "source_context_sha256": _sha(_wire(source)),
        "environment_sha256": _sha(_wire(environment)),
        "environment_ref_sha256": _sha(_wire(_environment_reference(environment))),
        "translation_sha256": _sha(_wire(translation))}


def persist_terminal_codebase_proof_index(*, source_bytes, source_path, decoder_experiment, qualification, output):
    """Reexecute native qualification with zero fitting, then persist exact keys."""
    import duckdb
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise ValueError("fresh canonical proof-index output required")
    source, decoder = _source_context(source_bytes, source_path, decoder_experiment)
    _artifact_receipts(qualification)
    environment = capture_terminal_codebase_proof_environment(
        lean_executable=qualification["lean_checks"][0]["executable"])
    translation = _translator()
    output.mkdir(parents=True, mode=0o700)
    replay = decoder_logic.qualify_terminal_codebase_decoder_logic(source_bytes=source_bytes,
        source_path=source_path, decoder=decoder, output=output / "checker-replay",
        lean_executable=environment["lean"]["executable"], z3_executable=environment["z3"]["executable"])
    if _wire(_qualification_semantics(replay)) != _wire(_qualification_semantics(qualification)):
        raise ValueError("supplied qualification differs from actual independent checker/inference replay")
    entries = _entries(source, replay, environment, translation)
    # Detect source, package and compiler drift across the real check invocation.
    after_source, _ = _source_context(source_bytes, source_path, decoder_experiment)
    if (after_source != source or _translator() != translation
            or capture_terminal_codebase_proof_environment(
                lean_executable=environment["lean"]["executable"],
                z3_executable=environment["z3"]["executable"]) != environment):
        raise ValueError("proof inputs changed during checker replay")
    database = output / "proof-index.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE model_evidence(entry_id VARCHAR PRIMARY KEY, relationship_id VARCHAR UNIQUE NOT NULL, key_json VARCHAR UNIQUE NOT NULL, row_json VARCHAR NOT NULL, row_sha256 VARCHAR NOT NULL)")
        for entry in entries:
            raw = _wire(entry)
            key = _wire(entry["key_relationship"]).decode()
            connection.execute("INSERT INTO model_evidence VALUES (?, ?, ?, ?, ?)",
                [entry["entry_id"], entry["key_relationship"]["relationship_id"], key, raw.decode(), _sha(raw)])
        connection.execute("CHECKPOINT")
    result = {"schema": SCHEMA, "output": str(output), "database": _file(database, MAX_DATABASE_BYTES),
        "source": source, "environment": environment, "translation": translation,
        "decoder_experiment": decoder_experiment,
        "source_snapshot": _source_snapshot(source, replay, environment, translation),
        "policy": POLICY, "entries": entries, "entry_count": len(entries),
        "replayed_qualification": replay, "historical_qualification_sha256": _sha(_wire(qualification)),
        "producer": "existing_native_decoder_and_header_checkers", "optimizer_steps": 0,
        "actual_solver_calls": replay["solver_calls"], "actual_lean_invocations": len(replay["lean_checks"]), **AUTHORITY}
    result["manifest_id"] = _identity(result)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(_wire(result) + b"\n")
    (output / "owner.lock").touch(exist_ok=False)
    return result


build_terminal_codebase_proof_index = persist_terminal_codebase_proof_index


def _verify_manifest(output, expected):
    output = Path(output).absolute()
    if output.resolve() != output or output.is_symlink() or expected.get("schema") != SCHEMA:
        raise ValueError("canonical externally pinned proof-index manifest required")
    payload = dict(expected)
    identifier = payload.pop("manifest_id", None)
    if identifier != _identity(payload) or expected["output"] != str(output):
        raise ValueError("proof-index manifest identity differs")
    if _read_json(output / "manifest.json") != expected:
        raise ValueError("persisted proof-index manifest differs from external pin")
    if (expected["policy"] != POLICY or expected["entry_count"] != 7
            or len(expected["entries"]) != 7 or len({row["entry_id"] for row in expected["entries"]}) != 7):
        raise ValueError("closed exact seven-entry profile required")
    if expected["source_snapshot"] != _source_snapshot(expected["source"],
            expected["replayed_qualification"], expected["environment"], expected["translation"]):
        raise ValueError("proof-index source snapshot projection differs")
    for key, value in AUTHORITY.items():
        if expected.get(key) is not value:
            raise ValueError("proof-index authority escalation rejected")
    if _file(output / "proof-index.duckdb", MAX_DATABASE_BYTES) != expected["database"]:
        raise ValueError("persisted native proof-index database differs")


def _database_entries(output, expected):
    import duckdb
    _verify_manifest(output, expected)
    with duckdb.connect(str(Path(output) / "proof-index.duckdb"), read_only=True) as connection:
        tables = connection.execute("SHOW TABLES").fetchall()
        if tables != [("model_evidence",)]:
            raise ValueError("proof-index native catalog differs")
        rows = connection.execute("SELECT entry_id, relationship_id, key_json, row_json, row_sha256 FROM model_evidence ORDER BY entry_id LIMIT 8").fetchall()
    if len(rows) != 7:
        raise ValueError("missing, duplicate or excess proof-index evidence")
    by_id = {row["entry_id"]: row for row in expected["entries"]}
    actual = []
    for entry_id, relationship_id, key_json, row_json, row_sha in rows:
        if len(row_json.encode()) > MAX_JSON_BYTES or len(key_json.encode()) > MAX_JSON_BYTES:
            raise ValueError("proof-index row budget exceeded")
        row, key = json.loads(row_json), json.loads(key_json)
        if (entry_id not in by_id or row != by_id[entry_id] or _sha(row_json.encode()) != row_sha
                or _wire(row).decode() != row_json or row["key_relationship"] != key
                or _wire(key).decode() != key_json or relationship_id != key["relationship_id"]
                or build_terminal_codebase_proof_key_relationship(dimensions=key["dimensions"]) != key):
            raise ValueError("native proof-index row/key integrity differs")
        actual.append(row)
    _verify_manifest(output, expected)
    return sorted(actual, key=lambda row: row["entry_id"])


def _lookup(entry, expected_environment):
    return {"schema": LOOKUP_SCHEMA, "status": "hit", "entry_id": entry["entry_id"],
        "key_relationship": entry["key_relationship"], "evidence": entry["evidence"],
        "expected_environment_sha256": _sha(_wire(expected_environment)), **AUTHORITY}


def lookup_terminal_codebase_model_evidence(*, output, expected, expected_key, expected_environment):
    """Native exact full-key equality; changed identity yields a bounded miss."""
    from ipfs_datasets_py.logic.common.canonical_cache_key import admit_cache_hit
    if expected_environment != expected["environment"]:
        raise ValueError("cross-environment proof-index lookup rejected")
    source, _ = _source_context(Path(expected["source"]["parent_capture"]["source"]["path"]).read_bytes(),
        expected["source"]["source_path"], expected["decoder_experiment"])
    current = capture_terminal_codebase_proof_environment(
        lean_executable=expected_environment["lean"]["executable"],
        z3_executable=expected_environment["z3"]["executable"])
    if (source != expected["source"] or current != expected_environment
            or _translator() != expected["translation"]):
        raise ValueError("current source/checkpoint/compiler/environment differs")
    _artifact_receipts(expected["replayed_qualification"])
    rebuilt = build_terminal_codebase_proof_key_relationship(dimensions=expected_key["dimensions"])
    if rebuilt != expected_key:
        raise ValueError("incomplete or forged proof-key relationship")
    with (Path(output) / "owner.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        entries = _database_entries(output, expected)
        import duckdb
        with duckdb.connect(str(Path(output) / "proof-index.duckdb"), read_only=True) as connection:
            rows = connection.execute("SELECT entry_id FROM model_evidence WHERE key_json = ? LIMIT 2",
                [_wire(expected_key).decode()]).fetchall()
        _verify_manifest(output, expected)
    if not rows:
        return {"schema": LOOKUP_SCHEMA, "status": "miss", "reason": "exact_key_absent",
            "key_relationship": expected_key, "evidence": None,
            "expected_environment_sha256": _sha(_wire(expected_environment)), **AUTHORITY}
    if len(rows) != 1:
        raise ValueError("ambiguous proof-index key")
    entry = next(row for row in entries if row["entry_id"] == rows[0][0])
    admit_cache_hit(entry["key_relationship"]["datasets_key"], expected_key["datasets_key"])
    return _lookup(entry, expected_environment)


def validate_terminal_codebase_model_evidence_lookup(*, lookup, expected_entry, expected_environment):
    """Pure complete-envelope admission for consumers with frozen native pins."""
    if lookup != _lookup(expected_entry, expected_environment):
        raise ValueError("conditional-model lookup differs from complete expected entry")
    relation = expected_entry["key_relationship"]
    if build_terminal_codebase_proof_key_relationship(dimensions=relation["dimensions"]) != relation:
        raise ValueError("conditional-model lookup key relation differs")
    if relation["dimensions"]["environment"] != _environment_reference(expected_environment):
        raise ValueError("conditional-model lookup environment differs")
    for key, value in AUTHORITY.items():
        if expected_entry["evidence"].get(key) is not value:
            raise ValueError("conditional-model evidence authority escalation")
    return lookup


def validate_terminal_codebase_proof_index(*, output, expected, source_bytes, source_path,
        decoder_experiment, qualification, fresh_process=False):
    """Rebind live source/checkpoint/compiler/environment and exact native rows.

    Checkers are not rerun on a cache read. The external manifest pins the actual
    producer's native receipts and artifacts; every current input is rehashed.
    """
    source, _ = _source_context(source_bytes, source_path, decoder_experiment)
    if source != expected["source"] or _translator() != expected["translation"]:
        raise ValueError("current source/checkpoint/compiler differs from proof-index pin")
    environment = capture_terminal_codebase_proof_environment(
        lean_executable=expected["environment"]["lean"]["executable"],
        z3_executable=expected["environment"]["z3"]["executable"])
    if environment != expected["environment"]:
        raise ValueError("current checker environment differs from proof-index pin")
    if _sha(_wire(qualification)) != expected["historical_qualification_sha256"]:
        raise ValueError("historical qualification reference differs")
    _artifact_receipts(expected["replayed_qualification"])
    if _entries(source, expected["replayed_qualification"], environment, expected["translation"]) != expected["entries"]:
        raise ValueError("stored native model evidence cannot be independently reconstructed")
    with (Path(output) / "owner.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        entries = _database_entries(output, expected)
    result = {"schema": "terminal-codebase-proof-index-validation@1", "manifest_id": expected["manifest_id"],
        "source_snapshot": expected["source_snapshot"],
        "entry_count": len(entries), "entries": [_lookup(entry, environment) for entry in entries],
        "optimizer_steps": 0, "checker_invocations_here": 0, "source_inference_replayed": True,
        "fresh_process_validated": False, **AUTHORITY}
    if fresh_process:
        request = {"output": str(output), "expected": expected, "source_path": source_path,
            "source_hex": source_bytes.hex(), "decoder_experiment": decoder_experiment,
            "qualification": qualification}
        completed = subprocess.run([sys.executable, "-m", __name__], input=_wire(request),
            capture_output=True, timeout=60, env=dict(os.environ))
        if completed.returncode != 0 or len(completed.stdout) > MAX_JSON_BYTES:
            raise ValueError("fresh native proof-index validation failed: " + completed.stderr.decode()[-2048:])
        independently = json.loads(completed.stdout)
        if independently != result:
            raise ValueError("fresh process exact proof-index reconstruction differs")
        result["fresh_process_validated"] = True
    return result


if __name__ == "__main__":
    request = json.loads(sys.stdin.buffer.read(MAX_JSON_BYTES + 1))
    request["source_bytes"] = bytes.fromhex(request.pop("source_hex"))
    result = validate_terminal_codebase_proof_index(**request)
    sys.stdout.buffer.write(_wire(result))


__all__ = ["persist_terminal_codebase_proof_index", "build_terminal_codebase_proof_index",
    "lookup_terminal_codebase_model_evidence", "validate_terminal_codebase_proof_index",
    "validate_terminal_codebase_model_evidence_lookup", "build_terminal_codebase_proof_key_relationship",
    "capture_terminal_codebase_proof_environment"]
