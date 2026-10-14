"""Complete native task projection and original compiler packet membership.

Native SQL/claim custody, source freshness and structural compilation are
separate checks. Neither this binding nor its receipt admits semantic intent,
worker execution, model ingestion or completion. Public task data omits only
explicit administrative timestamps and the top-level claim receipt.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref

from .current_source_task_admission import NativeSourceTaskAdmission
from .repository_world_worker_handoff import NativeRepositoryWorldWorkerHandoff
from .repository_world_packet import PreparedRepositoryWorldPacket, repository_world_task_binding, _plain
from .repository_context_compiler import RepositoryContextAdapter
from .repository_source_session import RepositorySourceSession
from ..context.context_compiler import ContextCompiler, ContextCompilationReceipt, ContextCompileResult
from ..context.context_contracts import ContextReference, ContextTier
from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity

PROJECTION_SCHEMA = "native-repository-world-task-projection@1"
PROJECTION_PROFILE = "complete_native_task_projection@1"
SCHEMA = "repository-world-compiled-worker-packet-binding@1"
PROFILE = "native_structural_readonly_packet@1"
MAX_ROWS_PER_RELATION = 64
MAX_TOTAL_RELATION_ROWS = 128
MAX_CELL_CHARACTERS = 16384
MAX_SOURCE_CHARACTERS = 65536
MAX_PUBLIC_TASK_BYTES = 32768
MAX_METADATA_BYTES = 16384
MAX_NODES = 16384
MAX_DEPTH = 24
FALSE_FIELDS = frozenset({"compiler_signature_verified", "semantic_alignment_verified", "world_dispatch_ready",
    "provider_launch", "provider_ingestion_verified", "proof_authority", "execution_authority",
    "completion_authority", "omission_authority", "worker_edit_authority", "model_tokens_measured"})
TASK_COLUMNS = ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "ordinal", "status",
    "revision", "priority", "created_at", "updated_at", "identity_json", "body_json", "extension_schema", "extension_json")
RELATIONS = {
    "dependencies": ("task_dependencies", ("task_cid", "dependency_task_cid", "kind"), "dependency_task_cid, kind"),
    "outputs": ("task_outputs", ("task_cid", "ordinal", "path", "effect_json"), "ordinal"),
    "acceptance": ("task_acceptance", ("task_cid", "ordinal", "criterion", "evidence_policy_json"), "ordinal"),
    "validations": ("task_validations", ("task_cid", "ordinal", "argv_json", "policy_json"), "ordinal"),
}
_SEAL = object()
_PROJECTIONS = weakref.WeakKeyDictionary()
_BINDINGS = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()
_ADMIN_OMISSIONS = ("task.created_at", "task.updated_at", "task.body.completion_receipt")


class RepositoryWorldWorkerPacketError(ValueError):
    """Closed errors without task text, SQL values, paths or credentials."""


def _need(value, code):
    if not value:
        raise RepositoryWorldWorkerPacketError(code)


def _structured(value, *, credential_keys=False):
    pending, count, chars = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= MAX_NODES and depth <= MAX_DEPTH, "worker_packet_structure_bound")
        if type(item) is dict:
            _need(len(item) <= MAX_NODES and all(type(k) is str for k in item), "worker_packet_structure_bound")
            for key, member in item.items():
                compact = re.sub(r"[^a-z0-9]", "", key.casefold())
                if credential_keys:
                    _need(not any(part in compact for part in ("credential", "secret", "token", "password", "privatekey", "apikey", "bearer"))
                          and compact not in {"authorizationheader", "authenticationheader"},
                          "worker_packet_credential_key_refused")
                pending.extend(((key, depth + 1), (member, depth + 1)))
        elif type(item) in {list, tuple}:
            _need(len(item) <= MAX_NODES, "worker_packet_structure_bound")
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is str:
            chars += len(item)
            _need(chars <= 4 * MAX_SOURCE_CHARACTERS, "worker_packet_text_bound")
        else:
            _need(item is None or type(item) is bool or type(item) is int and -(1 << 53) < item < (1 << 53),
                  "worker_packet_value_profile")


def _wire(value, *, maximum=MAX_METADATA_BYTES):
    _structured(value)
    raw = canonical_json_bytes(value)
    _need(len(raw) <= maximum, "worker_packet_byte_bound")
    return raw


def _json(raw):
    _need(type(raw) is str and len(raw) <= MAX_CELL_CHARACTERS, "worker_packet_cell_bound")
    def pairs(items):
        value = {}
        for key, member in items:
            _need(key not in value, "worker_packet_duplicate_json_key")
            value[key] = member
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _v: (_ for _ in ()).throw(RepositoryWorldWorkerPacketError("worker_packet_nonfinite_json")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise RepositoryWorldWorkerPacketError("worker_packet_invalid_json") from error
    _structured(value)
    return value


def _schema(connection, table, columns):
    rows = connection.execute("SELECT table_type FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?", [table]).fetchall()
    _need(len(rows) == 1 and rows[0][0] == "BASE TABLE", "worker_packet_native_table_kind")
    info = connection.execute("PRAGMA table_info('" + table + "')").fetchall()
    _need(tuple(row[1] for row in info) == columns and all(row[2] == ("BIGINT" if row[1] in {"ordinal", "revision"} else "VARCHAR") for row in info),
          "worker_packet_native_schema_changed")


def _population(task):
    """Only fixed original-owner SQL, bounded before materializing row text."""
    connection, task_id = task._connection, task._claim["task_cid"]
    specifications = [("task", "tasks", TASK_COLUMNS, "task_cid"),
        *((name, table, columns, ordering) for name, (table, columns, ordering) in RELATIONS.items())]
    totals, cells = 0, 0
    for name, table, columns, _ordering in specifications:
        _schema(connection, table, columns)
        text_columns = tuple(key for key in columns if key not in {"ordinal", "revision"})
        length = " + ".join("length(" + key + ")" for key in text_columns)
        aggregates = ["COUNT(*)", "COALESCE(SUM(" + length + "),0)"]
        aggregates += ["COALESCE(MAX(length(" + key + ")),0)" for key in text_columns]
        result = connection.execute("SELECT " + ",".join(aggregates) + " FROM " + table + " WHERE task_cid = ?", [task_id]).fetchone()
        _need(result is not None and len(result) == len(aggregates), "worker_packet_native_count_profile")
        # Native DuckDBRow iterates column names (Mapping), not cell values.
        # Read the fixed result geometry by ordinal without numeric coercion.
        result = tuple(result[index] for index in range(len(aggregates)))
        _need(all(type(v) is int and v >= 0 for v in result), "worker_packet_native_count_profile")
        _need(result[0] == 1 if name == "task" else result[0] <= MAX_ROWS_PER_RELATION, "worker_packet_relation_count_bound")
        _need(all(v <= MAX_CELL_CHARACTERS for v in result[2:]), "worker_packet_cell_bound")
        cells += result[1]
        totals += 0 if name == "task" else result[0]
        _need(cells <= MAX_SOURCE_CHARACTERS and totals <= MAX_TOTAL_RELATION_ROWS, "worker_packet_population_bound")
    result = {}
    for name, table, columns, ordering in specifications:
        rows = connection.execute("SELECT " + ",".join(columns) + " FROM " + table + " WHERE task_cid = ? ORDER BY " + ordering, [task_id]).fetchall()
        _need(len(rows) == 1 if name == "task" else len(rows) <= MAX_ROWS_PER_RELATION, "worker_packet_population_changed")
        values = []
        for row in rows:
            _need(len(row) == len(columns), "worker_packet_native_row_profile")
            row = tuple(row[index] for index in range(len(columns)))
            item = dict(zip(columns, row))
            _need(len(row) == len(columns) and item["task_cid"] == task_id
                  and all(type(item[key]) is int and item[key] >= 0 if key in {"ordinal", "revision"}
                          else type(item[key]) is str for key in columns), "worker_packet_native_row_profile")
            values.append(item)
        result[name] = values[0] if name == "task" else values
    _wire(result, maximum=4 * MAX_SOURCE_CHARACTERS)
    return result


def _public(population, native):
    task = dict(population["task"])
    body, identity, extension = (_json(task.pop(key)) for key in ("body_json", "identity_json", "extension_json"))
    _need(type(body) is dict and type(identity) is dict and type(extension) is dict,
          "worker_packet_native_object_profile")
    _need(content_identity(body) == native["expected_task_body_cid"] and task["revision"] == native["task_revision"],
          "worker_packet_native_body_binding")
    body = dict(body)
    _need("completion_receipt" in body, "worker_packet_native_claim_receipt_missing")
    body.pop("completion_receipt")
    task.pop("created_at")
    task.pop("updated_at")
    task.update(body=body, identity=identity, extension=extension,
        identity_json=population["task"]["identity_json"], extension_json=population["task"]["extension_json"])
    relations = {}
    for name in RELATIONS:
        values = []
        for row in population[name]:
            value = dict(row)
            for key in row:
                if key.endswith("_json"):
                    value[key.removesuffix("_json")] = _json(row[key])
            values.append(value)
        relations[name] = values
    _structured(task, credential_keys=True)
    _structured(relations, credential_keys=True)
    task_row_cid = content_identity(population["task"])
    relations_cid = content_identity({key: population[key] for key in RELATIONS})
    public = dict(schema=PROJECTION_SCHEMA, profile=PROJECTION_PROFILE, task=task, relations=relations,
        custody=dict(expected_task_body_cid=native["expected_task_body_cid"], native_task_row_cid=task_row_cid,
                     native_task_relations_cid=relations_cid), administrative_omissions=list(_ADMIN_OMISSIONS))
    return _wire(public, maximum=MAX_PUBLIC_TASK_BYTES), task_row_cid, relations_cid


class NativeRepositoryWorldTaskProjection:
    __slots__ = ("_seal", "_task", "_snapshot", "_population_json", "_text", "_metadata", "_lock", "_checking", "__weakref__")

    def __init__(self, seal, *, task_admission):
        _need(seal is _SEAL and type(self) is NativeRepositoryWorldTaskProjection
              and type(task_admission) is NativeSourceTaskAdmission, "worker_packet_original_native_task_required")
        native = NativeSourceTaskAdmission.require_current(task_admission)
        self._seal, self._task, self._snapshot = seal, task_admission, task_admission._snapshot
        self._lock, self._checking = threading.RLock(), False
        with task_admission._server._lock:
            NativeSourceTaskAdmission._native(task_admission)
            population = _population(task_admission)
        self._population_json = _wire(population, maximum=4 * MAX_SOURCE_CHARACTERS)
        self._text, row_cid, relations_cid = _public(population, native)
        metadata = dict(schema=PROJECTION_SCHEMA, profile=PROJECTION_PROFILE,
            task_cid=native["task_cid"], task_revision=native["task_revision"], expected_task_body_cid=native["expected_task_body_cid"],
            native_task_row_cid=row_cid, native_task_relations_cid=relations_cid,
            task_text_sha256=hashlib.sha256(self._text).hexdigest(), task_text_bytes=len(self._text),
            relation_counts={name: len(population[name]) for name in RELATIONS},
            native_validation_coverage="no_native_validation_commands_declared" if not population["validations"] else "declared_native_validation_commands_unexecuted",
            administrative_omissions=list(_ADMIN_OMISSIONS), native_task_relations_bound=True, **dict.fromkeys(FALSE_FIELDS, False))
        metadata["task_projection_cid"] = content_identity(metadata)
        self._metadata = _wire(metadata)

    def _seal_material(self):
        return (self._seal, self._task, self._snapshot, self._population_json, self._text, self._metadata, self._lock)

    def _retained(self):
        _need(type(self) is NativeRepositoryWorldTaskProjection and getattr(self, "_seal", None) is _SEAL,
              "worker_packet_original_projection_required")
        with _LOCK:
            issued = _PROJECTIONS.get(self)
            _need(issued is not None and all(a is b for a, b in zip(issued, NativeRepositoryWorldTaskProjection._seal_material(self))),
                  "worker_packet_projection_not_issued")
        _need(self._task._snapshot is self._snapshot, "worker_packet_native_snapshot_replaced")

    def require_current(self):
        NativeRepositoryWorldTaskProjection._retained(self)
        with self._lock:
            _need(not self._checking, "worker_packet_projection_reentry")
            self._checking = True
            try:
                native = NativeSourceTaskAdmission.require_current(self._task)
                with self._task._server._lock:
                    NativeSourceTaskAdmission._native(self._task)
                    population = _population(self._task)
                encoded = _wire(population, maximum=4 * MAX_SOURCE_CHARACTERS)
                public, _row, _relations = _public(population, native)
                _need(encoded == self._population_json and public == self._text, "worker_packet_native_population_changed")
                # Serialization and native owner callbacks precede the closing
                # original coordinator fence; its final detached SQL read has
                # no typed-client call or coordinator acquisition under SERVER.
                NativeSourceTaskAdmission.require_current(self._task)
                with self._task._server._lock:
                    NativeSourceTaskAdmission._native(self._task)
                    closing = _population(self._task)
                    _need(closing == population, "worker_packet_native_population_changed_after_fence")
                NativeRepositoryWorldTaskProjection._retained(self)
                return json.loads(self._metadata)
            finally:
                self._checking = False

    @property
    def task_text(self):
        NativeRepositoryWorldTaskProjection.require_current(self)
        return self._text.decode("utf-8", "strict")

    def public_task(self):
        NativeRepositoryWorldTaskProjection.require_current(self)
        return json.loads(self._text)

    def context_reference(self):
        metadata = NativeRepositoryWorldTaskProjection.require_current(self)
        return ContextReference(reference_id="native-task-projection:" + metadata["task_projection_cid"],
            kind="native-task-projection", tier=ContextTier.INVARIANT, referenced_content_id=metadata["task_projection_cid"],
            summary=self._text.decode(), metadata=dict(required=True, coverage_ids=(metadata["task_projection_cid"],)))

    def compile_core(self):
        metadata = NativeRepositoryWorldTaskProjection.require_current(self)
        public = json.loads(self._text)
        paths = tuple(row["path"] for row in public["relations"]["outputs"])
        _need(0 < len(paths) <= 64 and len(paths) == len(set(paths)), "worker_packet_native_output_scope_unavailable")
        effects = ("repository_read",)
        cid = metadata["task_projection_cid"]
        # Complete native business objects remain canonical DATA strings here.
        # The delta producer reserves structural keys such as "completion" for
        # authority; it must not interpret a native business field as authority.
        # The required original reference also retains the entire public object.
        return dict(goal=dict(native_task_contract_json=_wire(public["task"], maximum=MAX_PUBLIC_TASK_BYTES).decode(), native_task_projection_cid=cid),
            scope=dict(paths=paths, native_outputs_json=_wire(public["relations"]["outputs"], maximum=MAX_PUBLIC_TASK_BYTES).decode(),
                       native_dependencies_json=_wire(public["relations"]["dependencies"], maximum=MAX_PUBLIC_TASK_BYTES).decode(), native_task_projection_cid=cid),
            acceptance=dict(native_acceptance_json=_wire(public["relations"]["acceptance"], maximum=MAX_PUBLIC_TASK_BYTES).decode(),
                native_validations_json=_wire(public["relations"]["validations"], maximum=MAX_PUBLIC_TASK_BYTES).decode(), native_validation_coverage=metadata["native_validation_coverage"]),
            authority=dict(mode="readonly_observation", allowed_paths=paths, native_task_projection_cid=cid,
                world_task_packet=repository_world_task_binding(metadata["task_cid"], self._text.decode(), effects, paths)),
            allowed_effects=effects, write_paths=paths)


def capture_repository_world_task_projection(*, task_admission):
    projection = NativeRepositoryWorldTaskProjection(_SEAL, task_admission=task_admission)
    with _LOCK:
        _PROJECTIONS[projection] = NativeRepositoryWorldTaskProjection._seal_material(projection)
    NativeRepositoryWorldTaskProjection.require_current(projection)
    return projection


class NativeRepositoryWorldCompiledPacketBinding:
    """Exact original native projection, transitioned handoff and live packet.

    The compiler's task CID is a packet identity, never a native SQL task alias.
    The handoff binds the native public task bytes; the packet separately binds
    the compiled prompt produced on that same already-transitioned workspace.
    """

    __slots__ = ("_seal", "_projection", "_handoff", "_packet", "_task", "_adapter", "_session",
        "_compiler", "_compiler_refs", "_session_refs", "_session_bytes", "_packet_refs", "_compilation", "_metadata",
        "_lock", "_checking", "__weakref__")

    def __init__(self, seal, *, projection, handoff, packet):
        _need(seal is _SEAL and type(self) is NativeRepositoryWorldCompiledPacketBinding
              and type(projection) is NativeRepositoryWorldTaskProjection
              and type(handoff) is NativeRepositoryWorldWorkerHandoff
              and type(packet) is PreparedRepositoryWorldPacket, "worker_packet_original_owners_required")
        NativeRepositoryWorldTaskProjection.require_current(projection)
        NativeRepositoryWorldWorkerHandoff.require_current(handoff)
        PreparedRepositoryWorldPacket.validate_current(packet, workspace=handoff._workspace,
            task_text=projection._text.decode("utf-8", "strict"))
        self._seal, self._projection, self._handoff, self._packet = seal, projection, handoff, packet
        self._task = projection._task
        _need(self._task is handoff._task and packet._task_id == self._task._claim["task_cid"],
              "worker_packet_native_assignment_mismatch")
        self._adapter, self._session = packet._adapter, packet._adapter.session
        self._compiler = self._adapter.compiler
        _need(type(self._adapter) is RepositoryContextAdapter and type(self._session) is RepositorySourceSession
              and type(self._compiler) is ContextCompiler, "worker_packet_original_compiler_required")
        self._lock, self._checking = threading.RLock(), False
        self._packet_refs = PreparedRepositoryWorldPacket._seal(packet)
        self._compiler_refs = tuple((key, getattr(self._compiler, key)) for key in
            ("budget", "estimator", "value_policy", "provider_max_input_bytes", "budget_resolution",
             "effective_input_limit", "effective_budget"))
        self._compiler_refs += (("estimator._tokenizer", self._compiler.estimator._tokenizer),
                                ("estimator._chars_per_token", self._compiler.estimator._chars_per_token))
        self._session_refs = tuple((key, getattr(self._session, key)) for key in
            ("workspace", "state_dir", "session_id", "_index", "_manifest", "_capture_receipt",
             "_index_receipt", "_package", "_state_identity"))
        self._session_bytes = _wire(dict(manifest=self._session._manifest,
            capture_receipt=self._session._capture_receipt, index_receipt=self._session._index_receipt),
            maximum=4 * MAX_SOURCE_CHARACTERS)
        # Historical material supplies only the immutable original compilation
        # receipt. It cannot issue this live binding: packet issuance/currentness
        # and the same original compiler are independently checked below.
        from .repository_world_packet_replay import _decode_canonical
        raw = packet._replay_material_json.encode("utf-8", "strict")
        artifact = _decode_canonical(raw, hashlib.sha256(raw).hexdigest())
        receipt = ContextCompilationReceipt.from_dict(artifact["compilation_receipt"])
        self._compilation = ContextCompileResult(packet._capsule, receipt, receipt.decisions)
        _need(_plain(receipt.to_record()) == artifact["compilation_receipt"], "worker_packet_compilation_receipt_changed")
        self._metadata = None
        NativeRepositoryWorldCompiledPacketBinding._retained(self, issuing=True)
        metadata, _prompt = NativeRepositoryWorldCompiledPacketBinding._checked(self)
        metadata["binding_id"] = content_identity(metadata)
        self._metadata = _wire(metadata)

    def _seal_material(self):
        return (self._seal, self._projection, self._handoff, self._packet, self._task, self._adapter,
            self._session, self._compiler, self._compiler_refs, self._session_refs, self._packet_refs,
            self._session_bytes, self._compilation, self._metadata, self._lock)

    def _retained(self, *, issuing=False):
        _need(type(self) is NativeRepositoryWorldCompiledPacketBinding and getattr(self, "_seal", None) is _SEAL,
              "worker_packet_original_binding_required")
        if not issuing:
            with _LOCK:
                original = _BINDINGS.get(self)
                _need(original is not None, "worker_packet_binding_not_issued")
                current = NativeRepositoryWorldCompiledPacketBinding._seal_material(self)
                _need(len(original) == len(current) and all(a is b for a, b in zip(original, current)),
                      "worker_packet_original_binding_replaced")
        NativeRepositoryWorldTaskProjection._retained(self._projection)
        NativeRepositoryWorldWorkerHandoff._retained(self._handoff)
        _need(self._projection._task is self._task is self._handoff._task
              and self._packet._adapter is self._adapter and self._adapter.session is self._session
              and self._adapter.compiler is self._compiler, "worker_packet_original_owner_replaced")
        _need(all(a is b for a, b in zip(self._packet_refs, PreparedRepositoryWorldPacket._seal(self._packet))),
              "worker_packet_original_packet_replaced")
        from .repository_world_packet import _ISSUED, _ISSUANCE_LOCK
        with _ISSUANCE_LOCK:
            original_packet = _ISSUED.get(self._packet)
        _need(original_packet is not None and len(original_packet) == len(self._packet_refs)
              and all(a is b for a, b in zip(original_packet, self._packet_refs)),
              "worker_packet_original_issued_packet_replaced")
        _need(not any(key in self._packet.__dict__ for key in ("receipt", "validate_current", "_seal"))
              and not any(key in self._adapter.__dict__ for key in ("validate_current", "_observe", "_current", "_parent")),
              "worker_packet_original_owner_method_replaced")
        for key, original in self._compiler_refs:
            current = (getattr(self._compiler.estimator, key.split(".")[1]) if key.startswith("estimator.")
                       else getattr(self._compiler, key, None))
            _need(current is original, "worker_packet_original_compiler_replaced")
        _need(all(getattr(self._session, key, None) is original for key, original in self._session_refs),
              "worker_packet_original_source_session_replaced")
        _need(_wire(dict(manifest=self._session._manifest, capture_receipt=self._session._capture_receipt,
            index_receipt=self._session._index_receipt), maximum=4 * MAX_SOURCE_CHARACTERS) == self._session_bytes,
            "worker_packet_original_source_material_changed")

    def _checked(self):
        projection = NativeRepositoryWorldTaskProjection.require_current(self._projection)
        handoff = NativeRepositoryWorldWorkerHandoff.require_current(self._handoff)
        text = self._projection._text.decode("utf-8", "strict")
        _need(handoff["state"] == "TRANSITIONED" and handoff["read_only_transition_verified"] is True
              and (handoff["prompt_sha256"], handoff["prompt_bytes"])
                  == (projection["task_text_sha256"], projection["task_text_bytes"])
              and self._session.workspace == self._packet._workspace == self._handoff._workspace,
              "worker_packet_transitioned_task_workspace_mismatch")
        _need(not json.loads(self._projection._population_json)["validations"],
              "worker_packet_readonly_native_validations_unsupported")
        packet_receipt = PreparedRepositoryWorldPacket.validate_current(self._packet,
            workspace=self._handoff._workspace, task_text=text)
        _need(packet_receipt == PreparedRepositoryWorldPacket.receipt(self._packet),
              "worker_packet_original_receipt_changed")
        _need(packet_receipt.get("validation_profile") == "read_only_observation@1"
              and packet_receipt.get("native_validation_count") == 0
              and packet_receipt["validation_count"] == packet_receipt["proof_obligation_count"] == 0,
              "worker_packet_readonly_profile_required")
        ContextCompiler.verify_compile_result(self._compiler, self._compilation)
        core = NativeRepositoryWorldTaskProjection.compile_core(self._projection)
        capsule = self._packet._capsule
        for key in ("goal", "scope", "acceptance"):
            _need(_plain(getattr(capsule, key)) == _plain(core[key]), "worker_packet_native_core_mismatch")
        _need(_plain(capsule.authority.get("task")) == _plain(core["authority"]),
              "worker_packet_native_core_mismatch")
        expected = NativeRepositoryWorldTaskProjection.context_reference(self._projection)
        retained = tuple(ref for ref in (*capsule.evidence, *capsule.expansion_references)
                         if ref.kind == "native-task-projection")
        _need(len(retained) == 1 and retained[0] in capsule.evidence and retained[0].required
              and retained[0].canonical_bytes() == expected.canonical_bytes(),
              "worker_packet_complete_native_reference_missing")
        # The source session's full capture remains bound by its native owner;
        # additionally every original nominated/selected physical source must
        # occur with the same original bytes in that actual allocated capture.
        manifest = self._session._manifest
        _need(type(manifest) is dict and manifest.get("root_mode") == 0o750,
              "worker_packet_posttransition_source_capture_required")
        for raw_path, size, digest, _source_cid in self._handoff._source_rows:
            try:
                path = bytes.fromhex(raw_path).decode("utf-8", "strict")
            except (ValueError, UnicodeError) as error:
                raise RepositoryWorldWorkerPacketError("worker_packet_selected_path_profile") from error
            entry = manifest["entries"].get(path)
            _need(type(entry) is dict and entry.get("kind") == "file"
                  and entry.get("original_bytes") == size and entry.get("original_sha256") == digest,
                  "worker_packet_native_source_census_mismatch")
        prompt = self._packet.prompt
        metadata = dict(schema=SCHEMA, profile=PROFILE, packet_id=packet_receipt["packet_id"],
            context_cid=packet_receipt["context_cid"], prompt_sha256=packet_receipt["prompt_sha256"],
            prompt_bytes=packet_receipt["prompt_bytes"], handoff_id=handoff["handoff_id"],
            original_assignment_cid=handoff["original_assignment_cid"], task_cid=projection["task_cid"],
            task_revision=projection["task_revision"], expected_task_body_cid=projection["expected_task_body_cid"],
            task_projection_cid=projection["task_projection_cid"], native_task_row_cid=projection["native_task_row_cid"],
            native_task_relations_cid=projection["native_task_relations_cid"],
            compiled_task_packet_cid=packet_receipt["task_cid"],
            packet_receipt_sha256=hashlib.sha256(_wire(packet_receipt)).hexdigest(),
            source_generation=self._packet._generation,
            source_tree_sha256=packet_receipt["source_binding"]["source_tree_sha256"],
            structural_compiler_context_current=True, native_task_projection_current=True,
            native_assignment_continuity_verified=True, **dict.fromkeys(FALSE_FIELDS, False))
        if "intent_world" in packet_receipt:
            metadata["scoped_intent_receipt_sha256"] = hashlib.sha256(_wire(packet_receipt["intent_world"])).hexdigest()
        _wire(metadata)
        # Repeat original source/optional-intent owner fences after compiler
        # tokenizer callbacks. Final filesystem-only session checks below do
        # not invoke an index, tokenizer, CAS, or arbitrary callback.
        PreparedRepositoryWorldPacket.validate_current(self._packet,
            workspace=self._handoff._workspace, task_text=text)
        # Tokenizer/source/compiler/intent callbacks precede the original native
        # coordinator closing fence. No source callback follows the detached
        # final SQL/control/source checks under SERVER.
        NativeRepositoryWorldWorkerHandoff.require_current(self._handoff)
        NativeRepositoryWorldTaskProjection.require_current(self._projection)
        with self._task._server._lock:
            NativeRepositoryWorldWorkerHandoff._detached(self._handoff)
            _need(_wire(_population(self._task), maximum=4 * MAX_SOURCE_CHARACTERS)
                  == self._projection._population_json, "worker_packet_closing_native_population_changed")
            RepositorySourceSession._generation(self._session, self._packet._generation)
            RepositorySourceSession._current(self._session)
        _need(type(prompt) is str and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == metadata["prompt_sha256"]
              and len(prompt.encode("utf-8")) == metadata["prompt_bytes"], "worker_packet_prompt_changed")
        return metadata, prompt

    def require_current(self):
        NativeRepositoryWorldCompiledPacketBinding._retained(self)
        with self._lock:
            _need(not self._checking, "worker_packet_binding_reentry")
            self._checking = True
            try:
                metadata, _prompt = NativeRepositoryWorldCompiledPacketBinding._checked(self)
                metadata["binding_id"] = content_identity(metadata)
                _need(_wire(metadata) == self._metadata, "worker_packet_compiled_binding_changed")
                NativeRepositoryWorldCompiledPacketBinding._retained(self)
                return json.loads(self._metadata)
            finally:
                self._checking = False

    def require_for_delivery(self, *, handoff, original_prompt):
        NativeRepositoryWorldCompiledPacketBinding._retained(self)
        _need(type(original_prompt) in {str, bytes} and len(original_prompt) <= MAX_PUBLIC_TASK_BYTES,
              "worker_packet_delivery_assignment_mismatch")
        raw = original_prompt.encode("utf-8", "strict") if type(original_prompt) is str else original_prompt
        _need(handoff is self._handoff and raw == self._projection._text,
              "worker_packet_delivery_assignment_mismatch")
        return NativeRepositoryWorldCompiledPacketBinding.require_current(self)

    def delivery_material(self, *, handoff, original_prompt):
        with self._lock if getattr(self, "_lock", None) is not None else _LOCK:
            metadata = NativeRepositoryWorldCompiledPacketBinding.require_for_delivery(self,
                handoff=handoff, original_prompt=original_prompt)
            return metadata, self._packet.prompt


def capture_repository_world_compiled_packet_binding(*, projection, handoff, packet):
    binding = NativeRepositoryWorldCompiledPacketBinding(_SEAL, projection=projection, handoff=handoff, packet=packet)
    with _LOCK:
        _BINDINGS[binding] = NativeRepositoryWorldCompiledPacketBinding._seal_material(binding)
    NativeRepositoryWorldCompiledPacketBinding.require_current(binding)
    return binding
