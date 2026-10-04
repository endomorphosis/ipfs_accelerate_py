"""Isolated immutable current-source metadata and native relational joins.

Real existing AST, semantic graph, header-contract and lexical producers are
reconstructed from the complete supplied byte scope. These are structural or
candidate records, never software-behavior facts. The runner authenticates any
conditional proof-index producer before passing its externally pinned manifest.
No source is imported/executed and no training, provider or checker runs here.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

SCHEMA = "terminal-current-repository-index@1"
SOURCE_SCHEMA = "terminal-current-repository-source-snapshot@1"
BUILD_SCHEMA = "terminal-current-repository-generation@1"
QUERY_SCHEMA = "terminal-current-repository-query@1"
REQUIRED_PATHS = frozenset({"bottle.py", ".supervisor-instruction.md",
    ".supervisor-public-smoke.py", ".supervisor-authored-intent-control.md"})
LIMITS = {"source_bytes": 2_000_000, "total_source_bytes": 4_000_000,
    "json_bytes": 64 * 1024 * 1024, "json_nodes": 1_000_000, "json_depth": 48,
    "database_bytes": 256 * 1024 * 1024, "symbols": 2048,
    "ast_facts_per_file": 20000, "query_limit": 16, "restart_seconds": 60}
AUTHORITY = {"semantic_alignment_verified": False, "source_semantics_verified": False,
    "proof_authority": False, "execution_authority": False, "completion_authority": False,
    "mutation_authority": False, "behavioral_satisfaction": False}
POLICY = {"schema": "terminal-current-repository-policy@1",
    "source_scope": "exact_four_signed_development_sources",
    "invalidation": "any_source_task_or_producer_change_requires_new_generation",
    "retrieval": "complete_native_lexical_inventory_bounded_query_only",
    "proof_scope": "conditional_model_only",
    "live_authority_catalog": False, "learned_embeddings": False, **AUTHORITY}
_DDL = (
    "CREATE TABLE sources(path VARCHAR PRIMARY KEY,source_sha256 VARCHAR NOT NULL,source_bytes BIGINT NOT NULL,source_cid VARCHAR NOT NULL,generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL)",
    "CREATE TABLE symbols(stable_id VARCHAR PRIMARY KEY,path VARCHAR NOT NULL,qualified_symbol VARCHAR NOT NULL,local_symbol VARCHAR NOT NULL,version_cid VARCHAR NOT NULL,source_sha256 VARCHAR NOT NULL,source_cid VARCHAR NOT NULL,start_byte BIGINT,end_byte BIGINT,span_sha256 VARCHAR,source_ast_sha256 VARCHAR,kind VARCHAR NOT NULL,confidence VARCHAR NOT NULL,generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL)",
    "CREATE TABLE ast_facts(path VARCHAR NOT NULL,fact_id VARCHAR NOT NULL,source_sha256 VARCHAR NOT NULL,kind VARCHAR NOT NULL,name VARCHAR NOT NULL,owner VARCHAR NOT NULL,target VARCHAR NOT NULL,generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL,PRIMARY KEY(path,fact_id))",
    "CREATE TABLE kg_edges(edge_id VARCHAR PRIMARY KEY,source_id VARCHAR NOT NULL,target_id VARCHAR NOT NULL,relation VARCHAR NOT NULL,confidence VARCHAR NOT NULL,generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL)",
    "CREATE TABLE contracts(contract_id VARCHAR PRIMARY KEY,path VARCHAR,source_sha256 VARCHAR,property VARCHAR NOT NULL,candidate_applied BOOLEAN NOT NULL CHECK(candidate_applied=false),source_semantics_verified BOOLEAN NOT NULL CHECK(source_semantics_verified=false),generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL)",
    "CREATE TABLE lexical_vectors(row_id VARCHAR PRIMARY KEY,path VARCHAR NOT NULL,qualified_symbol VARCHAR NOT NULL,source_sha256 VARCHAR NOT NULL,index_id VARCHAR NOT NULL,config_id VARCHAR NOT NULL,model_id VARCHAR NOT NULL,dimensions INTEGER NOT NULL,embedding DOUBLE[] NOT NULL,generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL)",
    "CREATE TABLE conditional_proofs(entry_id VARCHAR NOT NULL,relationship_id VARCHAR NOT NULL,path VARCHAR NOT NULL,source_sha256 VARCHAR NOT NULL,stable_id VARCHAR NOT NULL,qualified_symbol VARCHAR NOT NULL,source_ast_sha256 VARCHAR NOT NULL,start_byte BIGINT NOT NULL,end_byte BIGINT NOT NULL,span_sha256 VARCHAR NOT NULL,property VARCHAR NOT NULL,model_domain VARCHAR NOT NULL,model_symbol VARCHAR NOT NULL,classification VARCHAR NOT NULL,checkpoint_sha256 VARCHAR NOT NULL,proof_index_manifest_id VARCHAR NOT NULL,source_semantics_verified BOOLEAN NOT NULL CHECK(source_semantics_verified=false),behavioral_satisfaction BOOLEAN NOT NULL CHECK(behavioral_satisfaction=false),generation_id VARCHAR NOT NULL,payload_json VARCHAR NOT NULL,PRIMARY KEY(entry_id,stable_id))",
    "CREATE TABLE repository_identity(singleton INTEGER PRIMARY KEY CHECK(singleton=1),identity_json VARCHAR NOT NULL)",
)
_TABLES = ("sources", "symbols", "ast_facts", "kg_edges", "contracts", "lexical_vectors", "conditional_proofs")


class RepositoryIndexError(ValueError):
    """Partial, changed, stale or forged repository generation."""


def _wire(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise RepositoryIndexError("finite bounded repository JSON required") from exc
    if len(raw) > LIMITS["json_bytes"]:
        raise RepositoryIndexError("repository JSON byte bound exceeded")
    return raw


def _json(value):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > LIMITS["json_nodes"] or depth > LIMITS["json_depth"]:
            raise RepositoryIndexError("repository JSON structure bound exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise RepositoryIndexError("exact string JSON object keys required")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise RepositoryIndexError("finite repository numbers required")
        elif type(item) not in (str, int, bool, type(None)):
            raise RepositoryIndexError("exact ordinary JSON values required")
    return json.loads(_wire(value))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(value):
    return "sha256:" + _sha(_wire(value))


def _object(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise RepositoryIndexError("exact " + name + " fields required")
    return value


def _text(value, name, maximum=4096):
    if (type(value) is not str or not value or value.strip() != value
            or len(value.encode()) > maximum or any(char in value for char in "\0\n\r")):
        raise RepositoryIndexError("bounded " + name + " required")
    return value


def _sources(source_bytes):
    if type(source_bytes) is not dict or set(source_bytes) != REQUIRED_PATHS:
        raise RepositoryIndexError("complete exact four-source scope required")
    total, result = 0, {}
    for path, raw in sorted(source_bytes.items()):
        if type(raw) is not bytes or not 0 < len(raw) <= LIMITS["source_bytes"]:
            raise RepositoryIndexError("bounded nonempty exact source bytes required")
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryIndexError("UTF-8 captured source required") from exc
        total += len(raw)
        result[path] = raw
    if total > LIMITS["total_source_bytes"]:
        raise RepositoryIndexError("repository source population bound exceeded")
    return result


def _file(path, maximum=LIMITS["json_bytes"]):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve() != path or info.st_size > maximum:
        raise RepositoryIndexError("bounded canonical regular artifact required")
    raw = path.read_bytes()
    after = path.lstat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) > maximum:
        raise RepositoryIndexError("artifact changed during read")
    return raw


def _pin(path, maximum=LIMITS["json_bytes"]):
    raw = _file(path, maximum)
    return {"relative_path": Path(path).name, "sha256": _sha(raw), "bytes": len(raw)}


def _write(path, value):
    raw = _wire(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o600)


def _producer():
    from ipfs_accelerate_py.agent_supervisor.analysis import program_ast_adapters, code_symbol_vector_index
    from ipfs_accelerate_py.agent_supervisor.core import conflict_graph
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_context_runtime
    from ipfs_datasets_py.logic.software_contracts.semantic_index import scanner
    from ipfs_datasets_py.logic.security_ir import doctor_header_contracts
    from . import terminal_codebase_supervisor_fixture

    modules = (program_ast_adapters, code_symbol_vector_index, conflict_graph,
        semantic_context_runtime, scanner, doctor_header_contracts, terminal_codebase_supervisor_fixture)
    paths = {Path(module.__file__).resolve() for module in modules} | {Path(__file__).resolve()}
    paths.update(Path(scanner.__file__).parent.glob("*.py"))
    if len(paths) > 128:
        raise RepositoryIndexError("producer inventory bound exceeded")
    return {"schema": "terminal-current-repository-producer@1", "python": sys.version,
        "files": [{"path": str(path), "sha256": _sha(_file(path, 4 * 1024 * 1024)),
            "bytes": path.stat().st_size} for path in sorted(paths)],
        "policy": POLICY}


def _source_units(state, sources):
    nodes, offsets = {}, {}
    for path, raw in sources.items():
        if not path.endswith(".py"):
            continue
        # Columns in native Python SourceSpan are UTF-8 byte offsets, not
        # Python string indexes. Preserve ordinary LF lines exactly.
        parts = raw.split(b"\n")
        lines = [part + b"\n" for part in parts[:-1]] + [parts[-1]]
        positions = [0]
        for line in lines:
            positions.append(positions[-1] + len(line))
        offsets[path] = positions
        index = {}
        for node in ast.walk(ast.parse(raw)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                key = (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
                index.setdefault(key, []).append(node)
        nodes[path] = index
    result = []
    for symbol in state.symbols:
        span, start, end, ast_sha = symbol.span, None, None, None
        if span is not None and span.start_line > 0 and span.end_line > 0:
            positions = offsets.get(symbol.module_path)
            if positions is not None:
                if span.start_line >= len(positions) or span.end_line >= len(positions):
                    raise RepositoryIndexError("native symbol span lies outside captured LF lines")
                start = positions[span.start_line - 1] + span.start_column
                end = positions[span.end_line - 1] + span.end_column
                raw = sources[symbol.module_path]
                if not 0 <= start <= end <= len(raw):
                    raise RepositoryIndexError("native symbol byte span lies outside current source")
                candidates = nodes[symbol.module_path].get((span.start_line, span.start_column,
                    span.end_line, span.end_column), [])
                if candidates:
                    # Function/class nodes precede descendants in ast.walk;
                    # header proof units therefore bind the original full AST.
                    ast_sha = _sha(ast.dump(candidates[0], include_attributes=False).encode())
        raw = sources[symbol.module_path]
        row = {"schema": "terminal-current-repository-source-unit@1", "stable_id": symbol.stable_id,
            "version_cid": symbol.version_cid, "path": symbol.module_path,
            "qualified_symbol": symbol.qualified_name,
            "local_symbol": symbol.qualified_name.rsplit(".", 1)[-1],
            "source_sha256": _sha(raw), "source_cid": symbol.source_cid,
            "span": None if span is None else span.to_dict(), "start_byte": start, "end_byte": end,
            "span_sha256": None if start is None else _sha(raw[start:end]), "source_ast_sha256": ast_sha,
            "kind": symbol.kind, "confidence": symbol.confidence}
        row["source_unit_id"] = _identity(row)
        result.append(row)
    return result


def build_current_repository_metadata(*, source_bytes, repository_id, task_spec):
    """Cold produce the complete current structural/candidate byte scope."""
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _scan_scoped_sources
    from ipfs_datasets_py.logic.security_ir.doctor_header_contracts import (
        WsgiHeaderProtocolContract, analyze_http_header_contracts,
    )
    from .terminal_codebase_supervisor_fixture import _replay_lexical_snapshot
    from .terminal_codebase_logic_qualification import REVIEW_PREMISE

    sources = _sources(source_bytes)
    repository_id = _text(repository_id, "explicit repository identity")
    task_spec = _json(task_spec)
    if type(task_spec) is not dict or not task_spec:
        raise RepositoryIndexError("complete independently declared task specification required")
    producer = _producer()
    state = _scan_scoped_sources(sources, repository_id=repository_id, max_symbols=LIMITS["symbols"])
    evidence = build_program_evidence_index({path: raw.decode() for path, raw in sources.items()},
        max_source_bytes=LIMITS["source_bytes"], max_facts=LIMITS["ast_facts_per_file"])
    if not evidence.exhaustive or evidence.truncated or {row.path for row in evidence.results} != set(sources):
        raise RepositoryIndexError("complete untruncated native AST source accounting required")
    python_sources = {path: raw for path, raw in sources.items() if path.endswith(".py")}
    python_evidence = build_program_evidence_index({path: raw.decode() for path, raw in python_sources.items()},
        max_source_bytes=LIMITS["source_bytes"], max_facts=LIMITS["ast_facts_per_file"])
    vectors = _replay_lexical_snapshot(evidence=python_evidence,
        source_hashes={path: _sha(raw) for path, raw in python_sources.items()})
    units = _source_units(state, sources)
    native_state = state.to_dict()
    snapshot_artifact = next(row for row in native_state["artifacts"]
        if row["artifact_id"] == "artifact:snapshot-evidence")
    source_cids = {row["path"]: row["source_cid"] for row in snapshot_artifact["metadata"]["snapshot"]["entries"]}
    leaves = [{"path": path, "sha256": _sha(raw), "bytes": len(raw), "source_cid": source_cids[path]}
        for path, raw in sources.items()]
    basis = {"schema": "terminal-current-repository-generation-basis@1", "repository_id": repository_id,
        "sources": leaves, "task_spec_sha256": _identity(task_spec), "producer_sha256": _identity(producer)}
    generation_id = _identity(basis)
    header = json.loads(_wire(analyze_http_header_contracts(sources["bottle.py"].decode(),
        protocol=WsgiHeaderProtocolContract(REVIEW_PREMISE)).to_dict()))
    facts = [{**fact.to_dict(), "path": row.path, "source_sha256": row.source_sha256}
        for row in evidence.results for fact in row.facts]
    records = {"sources": [{"path": path, "source_sha256": _sha(raw), "bytes": len(raw),
            "source_cid": source_cids[path], "source_text": raw.decode()} for path, raw in sources.items()],
        "ast": facts, "kg": [row.to_dict() for row in state.edges],
        "symbols": [row.to_dict() for row in state.symbols],
        "artifacts": [row.to_dict() for row in state.artifacts], "ast_indexes": [evidence.to_dict()],
        "retrieval_vectors": [row.to_dict() for row in vectors.rows],
        "contracts": [{"kind": "declared-authored-development-task", "task": task_spec, **AUTHORITY},
            {"kind": "source-bound-header-contract-analysis", "source_path": "bottle.py",
                "source_sha256": _sha(sources["bottle.py"]), "property": "header_delimiter_rejection",
                "analysis": header, "protocol_review_scope": "explicit_experiment_assumption",
                "candidate_applied": False, **AUTHORITY}],
        "repository_generation": [{"schema": BUILD_SCHEMA, "generation_id": generation_id,
            "basis": basis, "producer": producer, "task_spec": task_spec,
            "semantic_state": native_state, "vector_snapshot": vectors.to_dict(),
            "source_units": units, "python_source_hashes": {path: _sha(raw) for path, raw in python_sources.items()},
            "source_body_execution": False, "training_steps": 0, "checker_invocations": 0, **AUTHORITY}]}
    records = json.loads(_wire(records))
    source_snapshot = {"schema": SOURCE_SCHEMA, "generation_id": generation_id, "repository_id": repository_id,
        "sources": leaves, "task_spec_sha256": _identity(task_spec), "producer_sha256": _identity(producer),
        "semantic_state_cid": state.state_cid, "ast_index_id": evidence.ast_index.index_id,
        "lexical_index_id": vectors.index_id, "records_sha256": _identity(records),
        "complete_source_count": len(sources), "complete_python_source_count": len(python_sources), **AUTHORITY}
    source_snapshot["snapshot_id"] = _identity(source_snapshot)
    return _json({"records": records, "source_snapshot": source_snapshot})


def _checked_current(current):
    current = _json(current)
    _object(current, {"records", "source_snapshot"}, "current repository metadata")
    records = current["records"]
    expected_families = {"sources", "ast", "kg", "contracts", "symbols", "artifacts", "ast_indexes", "retrieval_vectors", "repository_generation"}
    _object(records, expected_families, "complete current record families")
    if any(type(rows) is not list for rows in records.values()) or len(records["repository_generation"]) != 1:
        raise RepositoryIndexError("complete current family arrays required")
    sources = {}
    for row in records["sources"]:
        if type(row) is not dict or type(row.get("source_text")) is not str or row.get("path") in sources:
            raise RepositoryIndexError("complete unique current source records required")
        sources[row["path"]] = row["source_text"].encode()
    snapshot = current["source_snapshot"]
    generation = records["repository_generation"][0]
    rebuilt = build_current_repository_metadata(source_bytes=sources,
        repository_id=snapshot.get("repository_id"), task_spec=generation.get("task_spec"))
    if rebuilt != current:
        raise RepositoryIndexError("current complete metadata differs from native producer reconstruction")
    return current


def _proof_rows(current, proof_index_manifest):
    if proof_index_manifest is None:
        return [], None
    from .terminal_codebase_proof_index import validate_terminal_codebase_model_evidence_lookup
    manifest = _json(proof_index_manifest)
    body = dict(manifest)
    identifier = body.pop("manifest_id", None)
    if (manifest.get("schema") != "terminal-codebase-proof-index@1" or identifier != _identity(body)
            or manifest.get("entry_count") != 7 or type(manifest.get("entries")) is not list
            or len(manifest["entries"]) != 7):
        raise RepositoryIndexError("complete externally pinned seven-entry conditional proof manifest required")
    for field in AUTHORITY:
        # Older native index manifests did not declare semantic alignment.
        # Its omission means unqualified; an explicit escalation still fails.
        value = manifest.get(field, False) if field == "semantic_alignment_verified" else manifest.get(field)
        if value is not False:
            raise RepositoryIndexError("conditional proof manifest authority escalation")
    sources = {row["path"]: row for row in current["records"]["sources"]}
    source = manifest["source"]
    if (source["source_path"] != "bottle.py" or source["source_sha256"] != sources["bottle.py"]["source_sha256"]
            or source["source_bytes"] != sources["bottle.py"]["bytes"]):
        raise RepositoryIndexError("conditional proof source leaf differs from current complete source")
    units = current["records"]["repository_generation"][0]["source_units"]
    by_header = {row["qualified_symbol"]: row for row in units if row["path"] == "bottle.py"}
    rows, seen = [], set()
    checkpoint = source["checkpoint"]["weights_sha256"]
    for entry in manifest["entries"]:
        entry_body = dict(entry)
        entry_id = entry_body.pop("entry_id", None)
        if entry_id != _identity(entry_body) or entry_id in seen:
            raise RepositoryIndexError("unique complete conditional proof entry identity required")
        seen.add(entry_id)
        lookup = {"schema": "terminal-codebase-model-evidence-lookup@1", "status": "hit",
            "entry_id": entry_id, "key_relationship": entry["key_relationship"], "evidence": entry["evidence"],
            "expected_environment_sha256": _sha(_wire(manifest["environment"]))}
        from .terminal_codebase_proof_index import AUTHORITY as MODEL_AUTHORITY
        lookup.update(MODEL_AUTHORITY)
        try:
            validate_terminal_codebase_model_evidence_lookup(lookup=lookup, expected_entry=entry,
                expected_environment=manifest["environment"])
        except (TypeError, ValueError) as exc:
            raise RepositoryIndexError("invalid complete conditional proof envelope: " + str(exc)) from exc
        evidence = entry["evidence"]
        if (evidence["source_path"] != "bottle.py" or evidence["source_sha256"] != source["source_sha256"]
                or entry["key_relationship"]["dimensions"]["source"] != source):
            raise RepositoryIndexError("conditional evidence source/checkpoint context differs")
        target = evidence["symbol"]
        if evidence["model_domain"] == "conditional_header_string_model":
            if target not in {"_hkey", "_hval"}:
                raise RepositoryIndexError("exact native SMT obligation target required")
            targets = {target}
        elif evidence["model_domain"] == "conditional_header_boolean_model":
            if target != "_hkey+_hval" or evidence["property"] != "header_boolean_model":
                raise RepositoryIndexError("exact joint conditional Boolean model target required")
            targets = {"_hkey", "_hval"}
        else:
            raise RepositoryIndexError("unsupported conditional model domain")
        footprint = set()
        for modeled in evidence["source_unit_bindings"]:
            if modeled["symbol"] in footprint:
                raise RepositoryIndexError("duplicate conditional model dependency footprint")
            footprint.add(modeled["symbol"])
            qualified = "bottle." + modeled["symbol"]
            unit = by_header.get(qualified)
            span = modeled["source_span"]
            if (unit is None or unit["source_ast_sha256"] != modeled["source_ast_sha256"]
                    or (unit["start_byte"], unit["end_byte"], unit["span_sha256"]) != (
                        span["start_byte"], span["end_byte"], span["sha256"])):
                raise RepositoryIndexError("conditional proof unit differs from exact current symbol/span/AST")
            # The source dependency footprint can be broader than the
            # obligation target. Only actual target units nominate evidence.
            if modeled["symbol"] not in targets:
                continue
            rows.append((entry_id, entry["key_relationship"]["relationship_id"], "bottle.py", source["source_sha256"],
                unit["stable_id"], qualified, unit["source_ast_sha256"], unit["start_byte"], unit["end_byte"],
                unit["span_sha256"], evidence["property"], evidence["model_domain"], target, evidence["classification"],
                checkpoint, identifier, False, False, current["source_snapshot"]["generation_id"], _wire(entry).decode()))
        if not targets <= footprint:
            raise RepositoryIndexError("conditional target is missing from its exact source dependency footprint")
    return sorted(rows), identifier


def _native_rows(current, proof_manifest):
    records, snapshot = current["records"], current["source_snapshot"]
    generation = snapshot["generation_id"]
    source_hashes = {row["path"]: row["source_sha256"] for row in records["sources"]}
    units = {row["stable_id"]: row for row in records["repository_generation"][0]["source_units"]}
    vector_snapshot = records["repository_generation"][0]["vector_snapshot"]
    config = vector_snapshot["config"]
    rows = {"sources": [(row["path"], row["source_sha256"], row["bytes"], row["source_cid"], generation,
        _wire(row).decode()) for row in records["sources"]], "symbols": [], "ast_facts": [], "kg_edges": [],
        "contracts": [], "lexical_vectors": []}
    for row in records["symbols"]:
        unit = units[row["stable_id"]]
        rows["symbols"].append((row["stable_id"], row["module_path"], row["qualified_name"], unit["local_symbol"],
            row["version_cid"], unit["source_sha256"], row["source_cid"], unit["start_byte"], unit["end_byte"],
            unit["span_sha256"], unit["source_ast_sha256"], row["kind"], row["confidence"], generation, _wire(row).decode()))
    for row in records["ast"]:
        rows["ast_facts"].append((row["path"], row["fact_id"], row["source_sha256"], row["kind"],
            row["name"], row["owner"], row["target"], generation, _wire(row).decode()))
    for row in records["kg"]:
        rows["kg_edges"].append((row["edge_id"], row["source_id"], row["target_id"], row["relation"],
            row["confidence"], generation, _wire(row).decode()))
    for row in records["contracts"]:
        rows["contracts"].append((_identity(row), row.get("source_path"), row.get("source_sha256"),
            row.get("property", "declared_task_spec"), False, False, generation, _wire(row).decode()))
    for row in records["retrieval_vectors"]:
        rows["lexical_vectors"].append((row["row_id"], row["path"], row["qualified_symbol"], source_hashes[row["path"]],
            vector_snapshot["index_id"], config["config_id"], config["model_id"], config["dimensions"],
            row["embedding"], generation, _wire(row).decode()))
    rows["conditional_proofs"], proof_id = _proof_rows(current, proof_manifest)
    return rows, proof_id


def _connect(path, read_only=False):
    import duckdb
    return duckdb.connect(str(path), read_only=read_only, config={"threads": 1,
        "autoinstall_known_extensions": "false", "autoload_known_extensions": "false"})


def _catalog(connection):
    """Pin real native types, names, constraints, views and explicit indexes."""
    return {
        "tables": connection.execute("SELECT schema_name,table_name,sql FROM duckdb_tables() WHERE NOT internal ORDER BY schema_name,table_name").fetchall(),
        "columns": connection.execute("SELECT schema_name,table_name,column_name,column_index,data_type,is_nullable,column_default FROM duckdb_columns() WHERE NOT internal ORDER BY schema_name,table_name,column_index").fetchall(),
        "constraints": connection.execute("SELECT schema_name,table_name,constraint_type,constraint_text,constraint_column_names FROM duckdb_constraints() ORDER BY schema_name,table_name,constraint_type,constraint_text").fetchall(),
        "views": connection.execute("SELECT schema_name,view_name,sql FROM duckdb_views() WHERE NOT internal ORDER BY schema_name,view_name").fetchall(),
        "indexes": connection.execute("SELECT schema_name,table_name,index_name,sql FROM duckdb_indexes() ORDER BY schema_name,table_name,index_name").fetchall(),
    }


def _reference_catalog():
    with _connect(":memory:") as connection:
        for ddl in _DDL:
            connection.execute(ddl)
        return _catalog(connection)


def _rows_digest(rows):
    return _identity({name: sorted(values, key=lambda row: _wire(row)) for name, values in rows.items()})


def persist_repository_index(*, current, proof_index_manifest=None, output):
    """Reconstruct producers, then seal an isolated native typed generation."""
    current = _checked_current(current)
    rows, proof_id = _native_rows(current, proof_index_manifest)
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.exists():
        raise RepositoryIndexError("fresh canonical external repository index output required")
    output.mkdir(parents=True, mode=0o700)
    lock_path = output / "owner.lock"
    lock_path.touch(exist_ok=False)
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        database = output / "repository-index.duckdb"
        identity = {"schema": SCHEMA, "source_snapshot": current["source_snapshot"],
            "proof_index_manifest_id": proof_id, "rows_sha256": _rows_digest(rows),
            "table_counts": {name: len(values) for name, values in rows.items()},
            "schema_sha256": _identity(list(_DDL)),
            "native_catalog_sha256": _identity(_reference_catalog()), "policy": POLICY}
        with _connect(database) as connection:
            connection.execute("BEGIN TRANSACTION")
            for ddl in _DDL:
                connection.execute(ddl)
            for name, values in rows.items():
                if values:
                    placeholders = ",".join("?" for _ in values[0])
                    connection.executemany("INSERT INTO " + name + " VALUES (" + placeholders + ")", values)
            connection.execute("INSERT INTO repository_identity VALUES (1,?)", [_wire(identity).decode()])
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        database.chmod(0o600)
        _write(output / "records.json", current["records"])
        result = {"schema": SCHEMA, "output": str(output), "identity": identity,
            "generation_id": current["source_snapshot"]["generation_id"],
            "source_snapshot": current["source_snapshot"], "proof_index_manifest_id": proof_id,
            "files": {"database": _pin(database, LIMITS["database_bytes"]), "records": _pin(output / "records.json")},
            "native_table_counts": identity["table_counts"], "complete_family_counts": {
                name: len(values) for name, values in current["records"].items()},
            "immutable": True, "native_typed_relational_columns": True, "native_ann_index": False,
            "native_graph_engine": False, "fresh_process_validated": False, **AUTHORITY}
        result["manifest_id"] = _identity(result)
        _write(output / "manifest.json", result)
    return result


def _validate(*, source_bytes, repository_id, task_spec, proof_index_manifest, expected, output):
    current = build_current_repository_metadata(source_bytes=source_bytes,
        repository_id=repository_id, task_spec=task_spec)
    expected = _json(expected)
    _object(expected, {"schema", "output", "identity", "generation_id", "source_snapshot", "proof_index_manifest_id",
        "files", "native_table_counts", "complete_family_counts", "immutable", "native_typed_relational_columns",
        "native_ann_index", "native_graph_engine", "fresh_process_validated", "manifest_id", *AUTHORITY},
        "externally pinned repository manifest")
    output = Path(output)
    body = dict(expected)
    manifest_id = body.pop("manifest_id", None)
    if (expected.get("schema") != SCHEMA or manifest_id != _identity(body)
            or expected.get("output") != str(output) or not output.is_absolute() or output.resolve() != output):
        raise RepositoryIndexError("canonical externally pinned repository manifest required")
    if (expected["source_snapshot"] != current["source_snapshot"]
            or expected["generation_id"] != current["source_snapshot"]["generation_id"]):
        raise RepositoryIndexError("current source/task/producer differs from immutable repository generation")
    if json.loads(_file(output / "manifest.json")) != expected:
        raise RepositoryIndexError("persisted repository manifest differs from external pin")
    for field in AUTHORITY:
        if expected.get(field) is not False:
            raise RepositoryIndexError("repository index authority escalation")
    rows, proof_id = _native_rows(current, proof_index_manifest)
    identity = {"schema": SCHEMA, "source_snapshot": current["source_snapshot"],
        "proof_index_manifest_id": proof_id, "rows_sha256": _rows_digest(rows),
        "table_counts": {name: len(values) for name, values in rows.items()},
        "schema_sha256": _identity(list(_DDL)),
        "native_catalog_sha256": _identity(_reference_catalog()), "policy": POLICY}
    if expected["identity"] != identity or expected["proof_index_manifest_id"] != proof_id:
        raise RepositoryIndexError("current complete native row/proof/schema identity differs")
    if (expected["native_table_counts"] != identity["table_counts"]
            or expected["complete_family_counts"] != {name: len(values) for name, values in current["records"].items()}
            or expected["immutable"] is not True or expected["native_typed_relational_columns"] is not True
            or any(expected[field] is not False for field in ("native_ann_index", "native_graph_engine", "fresh_process_validated"))):
        raise RepositoryIndexError("exact complete native repository qualification labels required")
    _object(expected["files"], {"database", "records"}, "sealed repository file inventory")
    for name, filename in (("database", "repository-index.duckdb"), ("records", "records.json")):
        pin = _object(expected["files"][name], {"relative_path", "sha256", "bytes"}, "sealed repository file pin")
        if pin["relative_path"] != filename:
            raise RepositoryIndexError("exact sealed repository artifact name required")
    with (output / "owner.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        if (_pin(output / "repository-index.duckdb", LIMITS["database_bytes"]) != expected["files"]["database"]
                or _pin(output / "records.json") != expected["files"]["records"]):
            raise RepositoryIndexError("native repository files differ from external pin")
        if json.loads(_file(output / "records.json")) != current["records"]:
            raise RepositoryIndexError("persisted complete records differ from current native producers")
        with _connect(output / "repository-index.duckdb", read_only=True) as connection:
            if _identity(_catalog(connection)) != identity["native_catalog_sha256"]:
                raise RepositoryIndexError("actual native repository catalog schema differs")
            if connection.execute("SHOW TABLES").fetchall() != [(name,) for name in sorted((*_TABLES, "repository_identity"))]:
                raise RepositoryIndexError("native repository catalog tables differ")
            if connection.execute("SELECT singleton,identity_json FROM repository_identity").fetchall() != [(1, _wire(identity).decode())]:
                raise RepositoryIndexError("native repository singleton identity differs")
            for name, values in rows.items():
                found = connection.execute("SELECT * FROM " + name + " LIMIT ?", [len(values) + 1]).fetchall()
                # Complete typed columns and payloads are checked, not only
                # convenient row-count/digest labels stored by the producer.
                if sorted(found, key=lambda row: _wire(row)) != sorted(values, key=lambda row: _wire(row)):
                    raise RepositoryIndexError("native complete row reconstruction differs: " + name)
        if _pin(output / "repository-index.duckdb", LIMITS["database_bytes"]) != expected["files"]["database"]:
            raise RepositoryIndexError("native repository changed during validation")
    return {"schema": "terminal-current-repository-index-validation@1", "status": "verified",
        "manifest_id": manifest_id, "generation_id": current["source_snapshot"]["generation_id"],
        "source_snapshot_id": current["source_snapshot"]["snapshot_id"],
        "native_table_counts": identity["table_counts"], "complete_source_count": len(source_bytes),
        "complete_native_producers_reconstructed": True, "fresh_process_validated": False,
        "training_steps": 0, "checker_invocations": 0, "source_body_execution": False, **AUTHORITY}, current


def validate_repository_index(*, source_bytes, repository_id, task_spec,
        proof_index_manifest=None, expected, output, fresh_process=False):
    """Read-only cold producer/SQL replay, optionally in an actual new process."""
    report, _ = _validate(source_bytes=source_bytes, repository_id=repository_id, task_spec=task_spec,
        proof_index_manifest=proof_index_manifest, expected=expected, output=output)
    if type(fresh_process) is not bool:
        raise RepositoryIndexError("exact restart Boolean required")
    if fresh_process:
        request = {"source_hex": {path: raw.hex() for path, raw in source_bytes.items()},
            "repository_id": repository_id, "task_spec": task_spec,
            "proof_index_manifest": proof_index_manifest, "expected": expected, "output": str(output)}
        completed = subprocess.run([sys.executable, "-m", __name__], input=_wire(request),
            capture_output=True, timeout=LIMITS["restart_seconds"], env=dict(os.environ))
        if completed.returncode != 0 or len(completed.stdout) > LIMITS["json_bytes"]:
            raise RepositoryIndexError("fresh native repository replay failed: " + completed.stderr.decode()[-2000:])
        if json.loads(completed.stdout) != report:
            raise RepositoryIndexError("fresh process repository reconstruction differs")
        report["fresh_process_validated"] = True
    return report


def query_repository_index(*, source_bytes, repository_id, task_spec,
        proof_index_manifest=None, expected, output, query):
    """Exact relational nominations plus bounded complete-inventory lexical search."""
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorQuery, search_code_symbol_vector_index,
    )
    report, current = _validate(source_bytes=source_bytes, repository_id=repository_id, task_spec=task_spec,
        proof_index_manifest=proof_index_manifest, expected=expected, output=output)
    query = _json(query)
    _object(query, {"source_path", "symbols", "property", "limit"}, "repository query")
    _text(query["source_path"], "query source path")
    _text(query["property"], "query property")
    if (type(query["symbols"]) is not list or len(query["symbols"]) > 16
            or any(type(symbol) is not str or not symbol for symbol in query["symbols"])
            or type(query["limit"]) is not int or not 1 <= query["limit"] <= LIMITS["query_limit"]):
        raise RepositoryIndexError("bounded exact symbol selection and query limit required")
    reasons = []
    if (query["source_path"] != "bottle.py" or query["property"] != "header_delimiter_rejection"
            or not query["symbols"] or not set(query["symbols"]) <= {"_hkey", "_hval"}):
        reasons.append("unsupported_reviewed_focus")
    if len(set(query["symbols"])) != len(query["symbols"]):
        reasons.append("ambiguous_symbol_selection")
    joins = []
    if not reasons:
        with (Path(output) / "owner.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            with _connect(Path(output) / "repository-index.duckdb", read_only=True) as connection:
                for symbol in sorted(query["symbols"]):
                    columns = connection.execute("""SELECT s.path,s.source_sha256,u.stable_id,u.qualified_symbol,u.version_cid,
                        u.start_byte,u.end_byte,u.span_sha256,u.source_ast_sha256,c.contract_id,
                        p.entry_id,p.relationship_id,p.property,p.model_domain,p.model_symbol,p.classification,p.checkpoint_sha256,
                        v.row_id,v.index_id,u.generation_id FROM sources s JOIN symbols u
                        ON u.path=s.path AND u.source_sha256=s.source_sha256 AND u.generation_id=s.generation_id
                        JOIN contracts c ON c.path=s.path AND c.source_sha256=s.source_sha256 AND c.generation_id=s.generation_id
                        JOIN conditional_proofs p ON p.stable_id=u.stable_id AND p.path=u.path AND p.source_sha256=u.source_sha256
                        AND p.start_byte=u.start_byte AND p.end_byte=u.end_byte AND p.span_sha256=u.span_sha256
                        AND p.source_ast_sha256=u.source_ast_sha256 AND p.generation_id=u.generation_id
                        LEFT JOIN lexical_vectors v ON v.path=u.path AND v.qualified_symbol=u.qualified_symbol
                        AND v.source_sha256=u.source_sha256 AND v.generation_id=u.generation_id
                        WHERE s.path=? AND u.qualified_symbol=? AND c.property=? AND p.model_symbol IN (?, '_hkey+_hval')
                        ORDER BY p.entry_id LIMIT 9""", [query["source_path"], "bottle." + symbol, query["property"], symbol]).fetchall()
                    for row in columns:
                        names = ("source_path", "source_sha256", "stable_symbol_id", "qualified_symbol", "symbol_version_cid",
                            "start_byte", "end_byte", "span_sha256", "source_ast_sha256", "contract_id",
                            "proof_entry_id", "proof_relationship_id", "proof_property", "model_domain", "model_symbol", "classification",
                            "checkpoint_sha256", "lexical_row_id", "lexical_index_id", "generation_id")
                        joins.append({**dict(zip(names, row, strict=True)), "runtime_refutation": False, **AUTHORITY})
            if _pin(Path(output) / "repository-index.duckdb", LIMITS["database_bytes"]) != expected["files"]["database"]:
                raise RepositoryIndexError("native repository changed during relational query")
    generation = current["records"]["repository_generation"][0]
    lexical = CodeVectorIndexSnapshot.from_dict(generation["vector_snapshot"])
    selected = [row for row in lexical.rows if row.path == query["source_path"]
        and row.qualified_symbol in {"bottle." + symbol for symbol in query["symbols"]}]
    ranked = None
    if selected and not reasons:
        vector = [sum(row.embedding[i] for row in selected) for i in range(lexical.config.dimensions)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            native_query = CodeVectorQuery.for_snapshot(lexical,
                query_vector=[value / norm for value in vector], max_results=query["limit"])
            # Native frozen vector result dictionaries contain tuple-valued
            # provenance fields. Serialize through canonical JSON exactly as
            # the native snapshots above; preserve every finite field.
            ranked = json.loads(_wire(search_code_symbol_vector_index(lexical, native_query).to_dict()))
    if not joins:
        reasons.append("no_exact_conditional_evidence_join_means_unknown")
    result = {"schema": QUERY_SCHEMA, "query": query,
        "query_id": _identity({"query": query, "generation_id": report["generation_id"],
            "proof_index_manifest_id": expected["proof_index_manifest_id"]}),
        "manifest_id": report["manifest_id"], "generation_id": report["generation_id"],
        "source_snapshot_id": report["source_snapshot_id"],
        "status": "nominated_conditional_model_only" if joins else "unknown", "joins": joins,
        "nominated_entry_ids": sorted({row["proof_entry_id"] for row in joins}),
        "ranked_lexical_context": ranked, "complete_lexical_inventory_count": len(lexical.rows),
        "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
        "reasons": [*reasons, "query_semantic_alignment_unproved", "request_domain_coverage_unproved",
            "conditional_model_source_semantics_unqualified"], **AUTHORITY}
    result["result_id"] = _identity({key: value for key, value in result.items() if key != "manifest_id"})
    return _json(result)


if __name__ == "__main__":
    request = json.loads(sys.stdin.buffer.read(LIMITS["json_bytes"] + 1))
    request["source_bytes"] = {path: bytes.fromhex(raw) for path, raw in request.pop("source_hex").items()}
    result = validate_repository_index(**request)
    sys.stdout.buffer.write(_wire(result))


__all__ = ["RepositoryIndexError", "SCHEMA", "SOURCE_SCHEMA", "BUILD_SCHEMA", "QUERY_SCHEMA",
    "build_current_repository_metadata", "persist_repository_index", "validate_repository_index",
    "query_repository_index"]
