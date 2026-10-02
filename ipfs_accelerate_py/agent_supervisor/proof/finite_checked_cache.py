"""Durable finite-table evidence, with fresh native checking on positive reuse.

The proof obligation is exactly the generated finite arithmetic table theorem
bundle. It is not Python runtime equivalence or a general program property.
Historical bodies and typed cache receipts cannot replace a fresh native call.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import tempfile
import time

from . import finite_cache_correspondence as correspondence
from .formal_verification_cache import FormalVerificationCache, ProofCacheKey
from .formal_verification_contracts import (
    CodeProofObligation, EvidenceAuthority, EvidenceKind, EvidenceVerdict,
    ProofEvidence, ProofReceipt, ProofVerdict, ResourceBudget, canonical_json, content_identity,
)
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as native
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)

SCHEMA = "finite-checked-cache-record@1"
MAX_BYTES = 4 * 1024 * 1024
NAMES = {"source": "captured_source.py", "compiled": "compiled.json", "driver": "driver.py",
    "request": "request.json", "tool_policy": "tool_policy.json", "trace": "observations.json",
    "python_process": "python_process.json", "lean_source": "FiniteInteger.lean",
    "lean_olean": "FiniteInteger.olean", "lean_certificate": "lean_certificate.json",
    "lean_process": "lean_process.json"}
SCOPE = "exact_generated_finite_integer_table_theorems_only"
# Correspondence pins the source/compiler/identity owners. These additional
# direct owners perform checking, resource admission and durable reconstruction.
EXECUTION_PRODUCERS = (
    __name__,
    "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state",
    "ipfs_accelerate_py.agent_supervisor.proof.proof_attestation",
    "ipfs_datasets_py.logic.backends.codebase_process",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.runtime_telemetry",
)
_TABLES = {
    "finite_checked_cache_meta": "schema_id VARCHAR PRIMARY KEY, schema_hash VARCHAR NOT NULL, catalog_hash VARCHAR NOT NULL, artifact_root VARCHAR NOT NULL",
    "finite_checked_cache_records": "request_key VARCHAR PRIMARY KEY, record_cid VARCHAR UNIQUE NOT NULL, status VARCHAR NOT NULL, formal_key_id VARCHAR NOT NULL",
    "finite_checked_cache_dependencies": "request_key VARCHAR NOT NULL, kind VARCHAR NOT NULL, value VARCHAR NOT NULL, PRIMARY KEY(request_key, kind, value)",
}
_COLUMNS = {
    "finite_checked_cache_meta": ("schema_id", "schema_hash", "catalog_hash", "artifact_root"),
    "finite_checked_cache_records": ("request_key", "record_cid", "status", "formal_key_id"),
    "finite_checked_cache_dependencies": ("request_key", "kind", "value"),
}


class FiniteCheckedCacheError(ValueError):
    """A durable finite record, current owner or native checker differs."""


def _require(value, reason):
    if not value:
        raise FiniteCheckedCacheError(reason)


def _raw(value):
    raw = canonical_dag_json_bytes(value)
    _require(len(raw) <= MAX_BYTES, "finite cache artifact exceeds byte bound")
    return raw


def _typed_payload(value):
    """Normalize reviewed supervisor enums before entering the native CAS."""
    return json.loads(canonical_json(value))


def _identity(value):
    return content_identity(value)


def _producer():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _execution_pins():
    return {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
            for name in EXECUTION_PRODUCERS}


def _status(observation):
    if observation["status"] != "observed":
        return observation["status"]
    return "positive" if observation["offset_clause_satisfied"] is True else "refuted"


def _dependencies(report):
    material = report["materials"]
    values = [("canonical:" + row["canonical_dimension"], report["canonical_key"][row["canonical_dimension"]])
              for row in correspondence._mapping()]
    values += [("source", material["source"]["source_cid"]),
               ("snapshot", material["source"]["head"]["snapshot_cid"]),
               ("contract", cid_for_structured(material["obligation"]["contract"])),
               ("model", cid_for_structured(material["source"]["model"]))]
    values += [("premise", cid_for_structured(p)) for p in material["assumptions"]]
    values += [("producer", name + ":sha256:" + digest)
               for name, digest in material["translation"]["producers"].items()]
    values += [("producer", name + ":sha256:" + digest) for name, digest in _execution_pins().items()]
    return sorted(set(values))


def _formal(report, observation, lean_source):
    """Derive a table-specific obligation and fully identified execution key."""
    material = report["materials"]
    premises = [{"id": _identity(value), "material": value} for value in material["assumptions"]]
    premise_ids = tuple(sorted(p["id"] for p in premises))
    tree = {"id": _identity(material["source"]), "material": material["source"]}
    goal = CodeProofObligation(
        repository_id=material["source"]["head"]["repository_id"], repository_tree_id=tree["id"],
        ast_scope_ids=(material["source"]["head"]["ast_revision_id"],),
        statement="The exact generated Lean finite table theorem bundle holds: " +
                  ", ".join(observation["lean_certificate"]["theorems"]),
        template_id=SCHEMA, template_version="1", template_semantic_hash="sha256:" + _producer(),
        invariant_class=SCOPE, premise_ids=premise_ids,
        metadata={"scope": SCOPE, "lean_source_cid": cid_for_bytes(lean_source),
                  "theorems": observation["lean_certificate"]["theorems"],
                  "domain_cid": observation["domain_cid"], "trace_cid": observation["trace_cid"],
                  "canonical_request": report["canonical_key"],
                  "source_runtime_semantics_verified": False, "execution_authority": False})
    components = {}
    for field, value in (("translator", {**material["translation"], "execution_producers": _execution_pins()}),
                         ("solver", {"role": "isolated_finite_observer", "python": material["environment"]["python"]}),
                         ("kernel", material["environment"]["lean"]), ("toolchain", material["environment"]),
                         ("theorem_registry", {"scope": SCOPE, "theorems": observation["lean_certificate"]["theorems"],
                                               "lean_source_cid": cid_for_bytes(lean_source)}),
                         ("policy", {"scope": SCOPE, "producer": _producer(), "native_policy": material["policy"],
                                     "positive_reuse_requires_fresh_native_observation": True})):
        components[field] = {"id": _identity(value), "material": value}
    policy = material["policy"]
    limits = policy["native_tools"]["process_limits"]
    budget = ResourceBudget(wall_time_ms=policy["operation_ceiling"]["timeout_seconds"] * 1000,
        memory_bytes=policy["operation_ceiling"]["memory_mb"] * 1024 * 1024,
        disk_bytes=limits["max_workspace_bytes"], max_processes=1, max_premises=len(premises),
        max_output_bytes=limits["max_output_bytes"], network_allowed=False)
    key = ProofCacheKey(obligation={"obligation_id": goal.obligation_id, "obligation": goal.to_dict(),
            "complete_correspondence": report}, premises=tuple(premises), **components,
            resource_budget=budget.to_dict(), candidate_tree=tree)
    return key, goal, budget


def _receipt(report, observation, bundle_cid, lean_source):
    key, goal, budget = _formal(report, observation, lean_source)
    certificate = observation["lean_certificate"]
    kernel_receipt = cid_for_structured(certificate)
    receipt = ProofReceipt(obligation_id=goal.obligation_id, plan_id=report["canonical_key_id"],
        attempt_id=observation["result_cid"], repository_id=goal.repository_id,
        repository_tree_id=goal.repository_tree_id, ast_scope_ids=goal.ast_scope_ids,
        premise_ids=goal.premise_ids, translator_id=key.translator["id"], solver_id=key.solver["id"],
        kernel_id=key.kernel["id"], toolchain_id=key.toolchain["id"], policy_id=key.policy["id"],
        theorem_registry_id=key.theorem_registry["id"], resource_budget=budget, verdict=ProofVerdict.PROVED,
        kernel_receipt_id=kernel_receipt,
        evidence=(ProofEvidence(kind=EvidenceKind.KERNEL_VERIFICATION, authority=EvidenceAuthority.KERNEL,
            verdict=EvidenceVerdict.ACCEPTED, artifact_id=kernel_receipt, subject_id=goal.obligation_id,
            verifier_id=key.kernel["id"], independent=True,
            metadata={"scope": SCOPE, "bundle_cid": bundle_cid,
                      "lean_source_cid": certificate["source_cid"], "olean_cid": certificate["olean_cid"]}),),
        metadata={"scope": SCOPE, "bundle_cid": bundle_cid, "native_observation_cid": observation["result_cid"],
                  "source_runtime_semantics_verified": False, "execution_authority": False,
                  "completion_authority": False, "fresh_checker_required_by_companion_owner": True})
    return key, receipt


class FiniteCheckedCache:
    """One finite evidence domain on the existing local cache owner and CAS."""

    def __init__(self, cache, artifacts, *, max_records=256):
        _require(type(cache) is FormalVerificationCache and type(artifacts) is ImmutableCAS,
                 "exact native formal cache and immutable artifact owners required")
        _require(type(max_records) is int and 1 <= max_records <= 4096, "bounded record capacity required")
        self.cache, self.artifacts, self.max_records = cache, artifacts, max_records
        with self._transaction(initialize=True) as cx:
            existing = {row[0] for row in cx.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_name IN (?, ?, ?) LIMIT 4",
                list(_TABLES)).fetchall()}
            selected = existing.intersection(_TABLES)
            _require(not selected or selected == set(_TABLES), "incomplete finite cache domain")
            if not selected:
                for name, columns in _TABLES.items():
                    cx.execute(f"CREATE TABLE {name} ({columns})")
                cx.execute("INSERT INTO finite_checked_cache_meta VALUES (?, ?, ?, ?)",
                    [SCHEMA, _identity(_TABLES), self._catalog_hash(cx), str(artifacts.root.resolve())])
            self._schema(cx)

    @contextmanager
    def _transaction(self, *, initialize=False):
        cx = self.cache._connect()
        try:
            _require(getattr(cx, "_transport_mode", "") != "quack" and not getattr(cx, "_default_catalog", None),
                     "finite cache requires the local serialized transactional owner")
            cx.execute("BEGIN TRANSACTION")
            try:
                if not initialize:
                    self._schema(cx)
                yield cx
                cx.execute("COMMIT")
            except BaseException:
                cx.execute("ROLLBACK")
                raise
        finally:
            cx.close()

    @staticmethod
    def _catalog_hash(cx):
        sizes = cx.execute("SELECT length(table_name),length(sql) FROM duckdb_tables() WHERE schema_name='main' AND table_name LIKE 'finite_checked_cache_%' LIMIT 4").fetchall()
        _require(len(sizes) == 3 and all(0 < row[0] <= 128 and 0 < row[1] <= 4096 for row in sizes),
                 "finite cache table inventory differs or exceeds bounds")
        rows = cx.execute("SELECT table_name, sql FROM duckdb_tables() WHERE schema_name='main' AND table_name LIKE 'finite_checked_cache_%' ORDER BY table_name LIMIT 4").fetchall()
        return _identity([[row[0], row[1]] for row in rows])

    @staticmethod
    def _normative_schema(cx):
        # Do not trust a caller-rehashed catalog metadata row. These projections
        # are independently fixed by the versioned owner, including uniqueness.
        sizes = cx.execute("SELECT length(column_name),length(data_type),coalesce(length(column_default),0) FROM duckdb_columns() WHERE schema_name='main' AND table_name IN (?, ?, ?) LIMIT 12", list(_TABLES)).fetchall()
        _require(len(sizes) == 11 and all(0 < row[0] <= 128 and 0 < row[1] <= 128 and 0 <= row[2] <= 1024 for row in sizes),
                 "finite cache normative column sizes differ")
        columns = cx.execute("SELECT table_name,column_name,column_index,data_type,is_nullable,column_default FROM duckdb_columns() WHERE schema_name='main' AND table_name IN (?, ?, ?) ORDER BY table_name,column_index LIMIT 12", list(_TABLES)).fetchall()
        expected = [[table, name, index, "VARCHAR", False, None]
                    for table in sorted(_COLUMNS) for index, name in enumerate(_COLUMNS[table], 1)]
        _require([[row[i] for i in range(6)] for row in columns] == expected,
                 "finite cache normative columns differ")
        sizes = cx.execute("SELECT length(constraint_type),length(to_json(constraint_column_names)),coalesce(length(expression),0) FROM duckdb_constraints() WHERE schema_name='main' AND table_name IN (?, ?, ?) LIMIT 16", list(_TABLES)).fetchall()
        _require(len(sizes) == 15 and all(0 < row[0] <= 32 and 0 < row[1] <= 512 and 0 <= row[2] <= 4096 for row in sizes),
                 "finite cache normative constraint sizes differ")
        constraints = cx.execute("SELECT table_name,constraint_type,constraint_column_names,expression FROM duckdb_constraints() WHERE schema_name='main' AND table_name IN (?, ?, ?) ORDER BY table_name,constraint_type,constraint_column_names LIMIT 16", list(_TABLES)).fetchall()
        expected = [[table, "NOT NULL", [name], None] for table in _COLUMNS for name in _COLUMNS[table]]
        expected += [["finite_checked_cache_meta", "PRIMARY KEY", ["schema_id"], None],
                     ["finite_checked_cache_records", "PRIMARY KEY", ["request_key"], None],
                     ["finite_checked_cache_records", "UNIQUE", ["record_cid"], None],
                     ["finite_checked_cache_dependencies", "PRIMARY KEY", ["request_key", "kind", "value"], None]]
        _require([[row[i] for i in range(4)] for row in constraints] == sorted(expected),
                 "finite cache normative constraints differ")

    def _schema(self, cx):
        self._normative_schema(cx)
        sizes = cx.execute("SELECT length(schema_id), length(schema_hash), length(catalog_hash), length(artifact_root) FROM finite_checked_cache_meta LIMIT 2").fetchall()
        _require(len(sizes) == 1 and all(type(sizes[0][i]) is int and 0 < sizes[0][i] <= 4096 for i in range(4)),
                 "finite cache metadata is missing, duplicated or oversized")
        rows = cx.execute("SELECT schema_id, schema_hash, catalog_hash, artifact_root FROM finite_checked_cache_meta LIMIT 2").fetchall()
        _require([tuple(row[i] for i in range(4)) for row in rows] == [(SCHEMA, _identity(_TABLES), self._catalog_hash(cx), str(self.artifacts.root.resolve()))],
                 "finite cache schema or artifact owner changed")
        count = cx.execute("SELECT count(*) FROM (SELECT 1 FROM finite_checked_cache_records LIMIT ?)",
                           [self.max_records + 1]).fetchone()[0]
        _require(count <= self.max_records, "finite cache capacity exceeded")

    def _blob(self, cid):
        path = self.artifacts.path_for(cid, source=True)
        _require(path.stat().st_size <= MAX_BYTES, "cached body exceeds byte bound")
        return self.artifacts.get_bytes(cid)

    def _json(self, cid):
        path = self.artifacts.path_for(cid)
        _require(path.stat().st_size <= MAX_BYTES, "cached record exceeds byte bound")
        value = self.artifacts.get(cid)
        _raw(value)
        _require(cid_for_structured(value) == cid, "cached record identity differs")
        return value

    def _bundle(self, observation):
        blobs = {}
        total = 0
        _require(set(observation["artifacts"]) <= set(NAMES), "unknown native artifact")
        for name, descriptor in observation["artifacts"].items():
            path = Path(descriptor["path"])
            _require(path.name == NAMES[name] and path.stat().st_size <= MAX_BYTES, "native artifact path/size differs")
            raw = path.read_bytes()
            total += len(raw)
            _require(total <= MAX_BYTES and len(raw) == descriptor["size_bytes"]
                     and hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
                     and cid_for_bytes(raw) == descriptor["cid"], "native artifact bytes differ")
            blobs[name] = self.artifacts.put_bytes(raw)
        value = {"schema": "finite-checked-artifact-bundle@1", "observation": observation, "body_cids": blobs}
        _raw(value)
        return self.artifacts.put(value)

    def _replay_bundle(self, cid, args):
        value = self._json(cid)
        _require(type(value) is dict and set(value) == {"schema", "observation", "body_cids"}
                 and value["schema"] == "finite-checked-artifact-bundle@1", "closed finite artifact bundle required")
        original = value["observation"]
        _require(original["result_cid"] == cid_for_structured({k: v for k, v in original.items() if k != "result_cid"}),
                 "original observation identity differs")
        _require(type(value["body_cids"]) is dict and set(value["body_cids"]) == set(original["artifacts"])
                 and set(value["body_cids"]) <= set(NAMES), "complete native artifact bodies required")
        view = deepcopy(original)
        blobs, total = {}, 0
        with tempfile.TemporaryDirectory(prefix="finite-cache-reconstruct-") as directory:
            root = Path(directory).resolve()
            view["output"] = str(root)
            for name, body_cid in value["body_cids"].items():
                raw = self._blob(body_cid)
                total += len(raw)
                _require(total <= MAX_BYTES and body_cid == original["artifacts"][name]["cid"], "artifact inventory differs")
                path = root / NAMES[name]
                path.write_bytes(raw)
                view["artifacts"][name]["path"] = str(path)
                blobs[name] = raw
            view["result_cid"] = cid_for_structured({k: v for k, v in view.items() if k != "result_cid"})
            (root / "result.json").write_bytes(_raw(view))
            native.validate_finite_integer_observation(view, expected_head=args["expected_head"],
                contract=args["contract"], inputs=args["inputs"], tool_policy=args["tool_policy"])
        return original, blobs

    def _read(self, request_key, args, report):
        with self._transaction() as cx:
            sizes = cx.execute("SELECT octet_length(encode(record_cid)), octet_length(encode(status)), octet_length(encode(formal_key_id)) FROM finite_checked_cache_records WHERE request_key=? LIMIT 2", [request_key]).fetchall()
            if not sizes:
                return None
            _require(len(sizes) == 1 and all(type(sizes[0][i]) is int and 0 <= sizes[0][i] <= 512 for i in range(3)), "duplicate or oversized finite cache row")
            row = cx.execute("SELECT record_cid, status, formal_key_id FROM finite_checked_cache_records WHERE request_key=? LIMIT 2", [request_key]).fetchone()
            record = self._json(row[0])
            _require(type(record) is dict and set(record) == {"schema", "request_key", "correspondence", "status", "bundle_cid", "formal_key", "receipt", "dependencies", "producer", "execution_producers", "scope"}
                and record["schema"] == SCHEMA and record["request_key"] == request_key
                and _raw(record["correspondence"]) == _raw(report) and record["producer"] == _producer()
                and record["execution_producers"] == _execution_pins()
                and record["scope"] == SCOPE and record["status"] == row[1], "finite cache record/source/producer differs")
            expected = _dependencies(report)
            sizes = cx.execute("SELECT length(kind), length(value) FROM finite_checked_cache_dependencies WHERE request_key=? LIMIT 129", [request_key]).fetchall()
            _require(len(sizes) <= 128 and all(0 < v[0] <= 64 and 0 < v[1] <= 1024 for v in sizes), "reverse dependency sizes exceed bounds")
            deps = cx.execute("SELECT kind, value FROM finite_checked_cache_dependencies WHERE request_key=? ORDER BY kind,value LIMIT 129", [request_key]).fetchall()
            _require(len(deps) <= 128 and [(v[0], v[1]) for v in deps] == expected and record["dependencies"] == [list(v) for v in expected], "finite reverse dependencies differ")
        observation, blobs = self._replay_bundle(record["bundle_cid"], args)
        _require(record["status"] == _status(observation), "negative/positive status differs from native evidence")
        if record["status"] == "positive":
            key, receipt = _receipt(report, observation, record["bundle_cid"], blobs["lean_source"])
            _require(_raw(key.to_dict()) == _raw(record["formal_key"]) and _raw(_typed_payload(receipt.to_dict())) == _raw(record["receipt"])
                     and key.key_id == row[2], "finite proof obligation/receipt differs")
            cached = self.cache.lookup(key)
            _require(cached.hit and cached.kernel_receipt.receipt_id == receipt.receipt_id,
                     "native typed proof cache reconstruction rejected finite receipt")
        else:
            _require(record["formal_key"] is None and record["receipt"] is None and row[2] == "", "negative record claims a positive receipt")
        return dict(record_cid=row[0], record=record, observation=observation, blobs=blobs)

    @contextmanager
    def _request(self, owner_inputs, timeout_seconds):
        _require(type(owner_inputs) is dict and owner_inputs["index"].artifacts is self.artifacts,
                 "source and cache must share the exact native artifact owner")
        _require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
                 and 0 < timeout_seconds <= 300, "bounded overall cache operation required")
        deadline = time.monotonic() + timeout_seconds
        pins = _execution_pins()
        args = dict(owner_inputs)
        def remaining():
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError
            event = args.get("cancel_event")
            if event is not None and event.is_set():
                raise LeaseCancelledError("finite cache operation cancelled")
            duration = deadline - time.monotonic()
            if duration <= 0:
                raise LeaseTimeoutError("finite cache operation expired")
            return duration
        report = correspondence.prepare_finite_cache_correspondence(**{**args, "timeout_seconds": remaining()})
        yield args, report, remaining
        correspondence.verify_finite_cache_correspondence(report, **{**args, "timeout_seconds": remaining()})
        _require(_execution_pins() == pins, "finite cache execution producers changed during operation")

    def _fresh(self, args, report, remaining):
        policy = report["materials"]["policy"]["operation_ceiling"]
        with tempfile.TemporaryDirectory(prefix="finite-cache-check-") as directory:
            kwargs = {name: args[name] for name in ("index", "repository", "expected_head", "contract", "inputs", "tool_policy")}
            kwargs.update({name: args[name] for name in ("scheduler", "parent_lease", "cancel_event") if name in args})
            observed = native.observe_finite_integer_source(**kwargs, output=Path(directory).resolve() / "native",
                timeout_seconds=min(policy["timeout_seconds"], remaining()), memory_mb=policy["memory_mb"])
            native.validate_finite_integer_observation(observed, expected_head=args["expected_head"],
                contract=args["contract"], inputs=args["inputs"], tool_policy=args["tool_policy"])
            bundle_cid = self._bundle(observed)
            original, blobs = self._replay_bundle(bundle_cid, args)
            material = report["materials"]
            _require(original["source_cid"] == material["source"]["source_cid"]
                and original["source_sha256"] == material["source"]["source_sha256"]
                and original["compiled_cid"] == material["translation"]["compiled_cid"], "fresh native observation differs from key source")
            return original, blobs, bundle_cid

    @staticmethod
    def _same_checked(stored, fresh, blobs):
        _require(_status(fresh) == stored["record"]["status"], "fresh checker disagrees with cached disposition")
        if fresh["status"] == "observed":
            _require(fresh["observations"] == stored["observation"]["observations"]
                and fresh["trace_cid"] == stored["observation"]["trace_cid"]
                and blobs["lean_source"] == stored["blobs"]["lean_source"], "fresh checker does not reproduce exact finite table theorem")

    @staticmethod
    def _result(stored, *, fresh):
        status = stored["record"]["status"]
        return dict(schema="finite-checked-cache-lookup@1", status=status,
            record_cid=stored["record_cid"], request_key=stored["record"]["request_key"],
            positive_reuse_eligible=bool(fresh and status == "positive"),
            fresh_native_observation=bool(fresh), scope=SCOPE,
            source_runtime_semantics_verified=False, execution_authority=False, completion_authority=False,
            evidence=stored["record"], historical_record_is_not_live_authority=True)

    def lookup(self, *, owner_inputs, timeout_seconds=120):
        """Exact bounded lookup; positive eligibility always executes fresh checks."""
        result = None
        with self._request(owner_inputs, timeout_seconds) as (args, report, remaining):
            stored = self._read(report["bridged_key_id"], args, report)
            if stored is None:
                result = dict(schema="finite-checked-cache-lookup@1", status="miss", positive_reuse_eligible=False)
            elif stored["record"]["status"] == "positive":
                fresh, blobs, _ = self._fresh(args, report, remaining)
                self._same_checked(stored, fresh, blobs)
                result = self._result(stored, fresh=True)
            else:
                result = self._result(stored, fresh=False)
        return result

    def check_and_store(self, *, owner_inputs, timeout_seconds=120):
        """Run native checks internally and publish immutable checked history."""
        with self._request(owner_inputs, timeout_seconds) as (args, report, remaining):
            request_key = report["bridged_key_id"]
            observed, blobs, bundle_cid = self._fresh(args, report, remaining)
            published = False

            def publish():
                # Coordinate only publication. Every caller has its own fresh
                # native check, and reconstructs both stores after the lease.
                nonlocal published
                remaining()
                existing = self._read(request_key, args, report)
                if existing is not None:
                    self._same_checked(existing, observed, blobs)
                    return {"record_cid": existing["record_cid"]}
                status = _status(observed)
                key, receipt = (_receipt(report, observed, bundle_cid, blobs["lean_source"])
                                if status == "positive" else (None, None))
                record = dict(schema=SCHEMA, request_key=request_key, correspondence=report, status=status,
                    bundle_cid=bundle_cid, formal_key=None if key is None else key.to_dict(),
                    receipt=None if receipt is None else _typed_payload(receipt.to_dict()), dependencies=[list(v) for v in _dependencies(report)],
                    producer=_producer(), execution_producers=_execution_pins(), scope=SCOPE)
                _raw(record)
                record_cid = self.artifacts.put(record)
                if receipt is not None:
                    _require(self.cache.put(key, receipt).stored, "native typed cache refused checked finite table")
                with self._transaction() as cx:
                    _require(cx.execute("SELECT 1 FROM finite_checked_cache_records WHERE request_key=? LIMIT 1", [request_key]).fetchone() is None,
                             "concurrent finite publication won; retry exact lookup")
                    count = cx.execute("SELECT count(*) FROM finite_checked_cache_records").fetchone()[0]
                    _require(count < self.max_records, "finite cache capacity exhausted")
                    cx.execute("INSERT INTO finite_checked_cache_records VALUES (?, ?, ?, ?)",
                        [request_key, record_cid, status, "" if key is None else key.key_id])
                    for kind, value in _dependencies(report):
                        cx.execute("INSERT INTO finite_checked_cache_dependencies VALUES (?, ?, ?)", [request_key, kind, value])
                stored = self._read(request_key, args, report)
                published = True
                return {"record_cid": stored["record_cid"]}

            outcome = self.cache.single_flight(ProofCacheKey.from_dict(report["bridged_key"]), publish,
                lease_seconds=300, wait_timeout_seconds=remaining())
            remaining()
            stored = self._read(request_key, args, report)
            _require(stored is not None and outcome == {"record_cid": stored["record_cid"]},
                     "publication coordination differs from reconstructed record")
            self._same_checked(stored, observed, blobs)
            result = self._result(stored, fresh=True)
            result["duplicate_publication"] = not published
        return result

    def dependents(self, kind, value, *, limit=32):
        """Bounded historical references only; each use must call exact lookup."""
        _require(type(kind) is str and type(value) is str and 0 < len(kind) <= 64 and 0 < len(value) <= 1024,
                 "bounded dependency selector required")
        _require(type(limit) is int and 1 <= limit <= 64, "bounded dependency result limit required")
        with self._transaction() as cx:
            sizes = cx.execute("SELECT length(d.request_key),length(r.record_cid) FROM finite_checked_cache_dependencies d LEFT JOIN finite_checked_cache_records r ON d.request_key=r.request_key WHERE d.kind=? AND d.value=? LIMIT ?", [kind,value,limit+1]).fetchall()
            _require(len(sizes) <= limit and all(type(v[i]) is int and 0 < v[i] <= 512 for v in sizes for i in range(2)), "dependency query overflows or has orphaned/oversized rows")
            rows = cx.execute("SELECT d.request_key,r.record_cid FROM finite_checked_cache_dependencies d LEFT JOIN finite_checked_cache_records r ON d.request_key=r.request_key WHERE d.kind=? AND d.value=? ORDER BY d.request_key LIMIT ?", [kind,value,limit+1]).fetchall()
            _require(len(rows) <= limit and all(row[1] is not None for row in rows), "dependency query overflows or has orphaned rows")
            for row in rows:
                record = self._json(row[1])
                _require(record.get("schema") == SCHEMA and record.get("request_key") == row[0]
                         and [kind,value] in record.get("dependencies", []), "dependency does not join exact immutable record")
            return [{"request_key": row[0], "record_cid": row[1], "historical_only": True, "positive_reuse_eligible": False} for row in rows]


__all__ = ["FiniteCheckedCache", "FiniteCheckedCacheError", "SCHEMA", "SCOPE", "EXECUTION_PRODUCERS"]
