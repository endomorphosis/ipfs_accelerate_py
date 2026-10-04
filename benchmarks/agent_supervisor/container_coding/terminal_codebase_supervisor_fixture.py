"""Private, source-preserving Supervisor preparation for a codebase experiment.

This invokes the existing public preparation and real initial-context trainer.
It stops before planning, admission or workers. Metadata exports retain complete
producer inventories; bounded worker nominations are a separate projection.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time


SCHEMA = "terminal-codebase-supervisor-fixture@1"
MAX_SOURCE_BYTES = 2_000_000
MAX_INSTRUCTION_BYTES = 32_768
MAX_ARTIFACT_BYTES = 32_000_000
_AUTHORITY = {"proof_authority": False, "execution_authority": False,
              "completion_authority": False, "official_reward": None}
ARTIFACT_SCHEMA = "terminal-codebase-metadata-artifact@1"
CHUNK_SCHEMA = "terminal-codebase-metadata-chunk@1"
CHUNK_BYTES = 49_152
CHUNK_FAMILY = "metadata_chunks"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read(path: Path, limit: int) -> bytes:
    path = Path(path).absolute()
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.resolve(strict=True) != path
            or before.st_size > limit):
        raise ValueError("bounded canonical regular input required: " + str(path))
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during capture: " + str(path))
    if len(raw) > limit:
        raise ValueError("input exceeds byte bound")
    return raw


def _write(path: Path, value) -> dict:
    raw = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o600)
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repository), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


def _replay_lexical_snapshot(*, evidence, source_hashes: dict):
    """Rebuild the deterministic existing lexical producer from exact ASTs."""
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import build_code_symbol_vector_index
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity

    ast, docs = evidence.ast_index, {}
    if (not evidence.exhaustive or {item.path for item in evidence.results} != set(source_hashes)
            or any(item.status != "success" for item in evidence.results)
            or {item.path for item in ast.path_records} != set(source_hashes)):
        raise ValueError("replayed native AST coverage is incomplete")
    for indexed in ast.path_records:
        parts = list(PurePosixPath(indexed.path.removesuffix(".py")).parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        for symbol in indexed.ast_record.qualified_symbols:
            qualified = module + "." + symbol if module else symbol
            docs[qualified] = Counter(re.findall("[a-z0-9]+", qualified.lower()))
    if not docs:
        raise ValueError("replayed native AST has no qualified symbols")
    vocabulary = sorted({term for doc in docs.values() for term in doc})
    weights = {term: 1 + math.log((1 + len(docs)) / (1 + sum(term in doc for doc in docs.values())))
               for term in vocabulary}

    def vector(row):
        words = Counter(re.findall("[a-z0-9]+", row.qualified_symbol.lower()))
        values = [words[term] * weights[term] for term in vocabulary]
        norm = math.sqrt(sum(value * value for value in values))
        if not norm:
            raise ValueError("replayed lexical symbol has no vocabulary terms")
        return [value / norm for value in values]

    scope = content_identity({"schema": "permitted-vector-inputs@1", "sources": source_hashes})
    config = content_identity({"vocabulary": vocabulary,
                              "weights": {term: value.hex() for term, value in weights.items()}})
    return build_code_symbol_vector_index(ast, forest_id=scope, tree_id=scope,
        coverage_id=ast.index_id, dimensions=len(vocabulary), model_id="lexical-tfidf-symbols@1",
        model_revision="1", configuration_id=config, vectors=vector)


def bound_terminal_codebase_metadata_records(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Preserve whole producer rows within existing native metadata bounds.

    Large containers use exact canonical JSON chunks rather than truncated
    inventories or increased storage limits. Individual supported rows retain
    their existing form. Applying this to an already bounded export is stable.
    """
    from .codebase_ir_metadata import LIMITS, MetadataError, _plain

    if (type(records) is not dict or any(type(name) is not str for name in records)
            or any(type(rows) is not list for rows in records.values())):
        raise ValueError("metadata family arrays required")
    if CHUNK_FAMILY in records:
        records = reconstruct_terminal_codebase_metadata_records(records)
    result, chunks = {}, []
    for family, rows in records.items():
        result[family] = []
        for ordinal, row in enumerate(rows):
            if type(row) is not dict or row.get("schema") in (ARTIFACT_SCHEMA, CHUNK_SCHEMA):
                raise ValueError("ordinary producer object row required")
            raw = _json_bytes(row)
            if len(raw) > MAX_ARTIFACT_BYTES:
                raise ValueError("complete metadata artifact exceeds experiment byte bound")
            # Existing native to_dict records can retain tuple fields. Preserve
            # their exact JSON representation as ordinary finite wire values.
            wire_row = json.loads(raw)
            try:
                _plain(wire_row)
                bounded = len(raw) <= LIMITS["row_bytes"]
            except MetadataError:
                bounded = False
            if bounded:
                result[family].append(wire_row)
                continue
            parts = [raw[start:start + CHUNK_BYTES] for start in range(0, len(raw), CHUNK_BYTES)]
            descriptor = {"schema": ARTIFACT_SCHEMA, "family": family,
                "original_ordinal": ordinal, "payload_sha256": _sha(raw),
                "payload_bytes": len(raw), "chunk_count": len(parts),
                "encoding": "canonical-json-utf8/base64-chunks"}
            artifact_id = "sha256:" + _sha(_json_bytes(descriptor))
            result[family].append({**descriptor, "artifact_id": artifact_id})
            for part_ordinal, part in enumerate(parts):
                chunk = {"schema": CHUNK_SCHEMA, "artifact_id": artifact_id,
                    "ordinal": part_ordinal, "sha256": _sha(part), "bytes": len(part),
                    "base64": base64.b64encode(part).decode("ascii")}
                chunks.append({**chunk, "chunk_id": "sha256:" + _sha(_json_bytes(chunk))})
    if chunks:
        result[CHUNK_FAMILY] = chunks
    return result


def reconstruct_terminal_codebase_metadata_records(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Recover complete producer rows from an export, rejecting broken links."""
    if type(records) is not dict or any(type(rows) is not list for rows in records.values()):
        raise ValueError("metadata family arrays required")
    chunks_by_artifact, encountered, consumed = {}, set(), set()
    for chunk in records.get(CHUNK_FAMILY, []):
        keys = {"schema", "artifact_id", "ordinal", "sha256", "bytes", "base64", "chunk_id"}
        if (type(chunk) is not dict or set(chunk) != keys or chunk["schema"] != CHUNK_SCHEMA
                or type(chunk["ordinal"]) is not int or chunk["ordinal"] < 0
                or type(chunk["bytes"]) is not int or not 1 <= chunk["bytes"] <= CHUNK_BYTES):
            raise ValueError("exact bounded metadata chunk required")
        body = {key: value for key, value in chunk.items() if key != "chunk_id"}
        if chunk["chunk_id"] != "sha256:" + _sha(_json_bytes(body)):
            raise ValueError("metadata chunk identity differs")
        try:
            raw = base64.b64decode(chunk["base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid metadata chunk encoding") from exc
        if len(raw) != chunk["bytes"] or _sha(raw) != chunk["sha256"]:
            raise ValueError("metadata chunk bytes differ")
        if chunk["chunk_id"] in encountered:
            raise ValueError("duplicate metadata chunk")
        encountered.add(chunk["chunk_id"])
        rows = chunks_by_artifact.setdefault(chunk["artifact_id"], [])
        if chunk["ordinal"] != len(rows):
            raise ValueError("metadata chunk order differs")
        rows.append(raw)
    result = {}
    for family, rows in records.items():
        if family == CHUNK_FAMILY:
            continue
        result[family] = []
        for ordinal, row in enumerate(rows):
            if type(row) is not dict:
                raise ValueError("metadata object row required")
            if row.get("schema") != ARTIFACT_SCHEMA:
                result[family].append(row)
                continue
            keys = {"schema", "family", "original_ordinal", "payload_sha256", "payload_bytes",
                    "chunk_count", "encoding", "artifact_id"}
            if (set(row) != keys or row["family"] != family or row["original_ordinal"] != ordinal
                    or type(row["payload_bytes"]) is not int or row["payload_bytes"] < 1
                    or type(row["chunk_count"]) is not int or row["chunk_count"] < 1
                    or row["encoding"] != "canonical-json-utf8/base64-chunks"):
                raise ValueError("exact metadata artifact descriptor required")
            body = {key: value for key, value in row.items() if key != "artifact_id"}
            artifact_id = "sha256:" + _sha(_json_bytes(body))
            if artifact_id != row["artifact_id"] or artifact_id in consumed:
                raise ValueError("metadata artifact identity differs")
            parts = chunks_by_artifact.get(artifact_id, [])
            raw = b"".join(parts)
            if (len(parts) != row["chunk_count"] or len(raw) != row["payload_bytes"]
                    or _sha(raw) != row["payload_sha256"]):
                raise ValueError("metadata artifact payload differs")
            try:
                original = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("invalid metadata artifact JSON") from exc
            if type(original) is not dict or _json_bytes(original) != raw:
                raise ValueError("metadata artifact is not canonical object JSON")
            consumed.add(artifact_id)
            result[family].append(original)
    if consumed != set(chunks_by_artifact):
        raise ValueError("orphan metadata chunks")
    return result


def prepare_terminal_codebase_fixture(*, source: Path, instruction: Path,
                                     output: Path) -> dict:
    """Copy exact permitted Bottle inputs into a fresh private experiment.

    The caller supplies public pre-edit source provenance. No benchmark root,
    hidden verifier, historical solution, provider or existing state is loaded.
    Only Bottle is initially committed; public preparation adds its instruction
    and structural smoke check. The trainer also sees that declared smoke file.
    """
    from . import terminal_indexed_preparation as prep
    from .terminal_initial_context import prepare_initial_context

    source, instruction, output = (Path(value).absolute()
                                   for value in (source, instruction, output))
    if source.name != "bottle.py":
        raise ValueError("the supported terminal fixture requires bottle.py")
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise ValueError("a fresh canonical output directory is required")
    if source.is_relative_to(output) or instruction.is_relative_to(output):
        raise ValueError("experiment outputs must be separate from original inputs")
    raw = _read(source, MAX_SOURCE_BYTES)
    public = _read(instruction, MAX_INSTRUCTION_BYTES)
    if not public.decode("utf-8").strip():
        raise ValueError("nonempty UTF-8 public instruction required")
    raw.decode("utf-8")
    source_mode = stat.S_IMODE(source.stat().st_mode)
    started = time.monotonic()
    output.mkdir(mode=0o700)
    repository, state = output / "repository", output / "state"
    repository.mkdir(mode=0o700)
    (repository / "bottle.py").write_bytes(raw)
    (repository / "bottle.py").chmod(source_mode)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Bounded source experiment")
    _git(repository, "config", "user.email", "experiment@example.invalid")
    _git(repository, "add", "--", "bottle.py")
    _git(repository, "commit", "-qm", "Capture exact public pre-edit Bottle bytes")
    captured_commit = _git(repository, "rev-parse", "HEAD")
    prepared = prep.prepare(repository=repository, instruction=instruction, state=state,
                            disable_intent_autoencoder=True)
    # Full Bottle is retained in the source CAS and metadata. Inlining all source
    # would exceed the existing 32768-byte worker projection; do not relax it.
    initial = prepare_initial_context(state=state, prepared=prepared,
        model_snapshot=None, model_revision="", train_autoencoder=True,
        required_raw_paths=[prep.INSTRUCTION, prep.SMOKE])
    if (initial["provider_calls"] != 0 or initial["canonical_tasks_created"] is not False
            or initial["world_task_count"] != 0):
        raise ValueError("fixture crossed the preplanning preparation boundary")
    if (_read(source, MAX_SOURCE_BYTES) != raw
            or stat.S_IMODE(source.stat().st_mode) != source_mode
            or _read(instruction, MAX_INSTRUCTION_BYTES) != public
            or _read(repository / "bottle.py", MAX_SOURCE_BYTES) != raw
            or _read(repository / prep.INSTRUCTION, MAX_INSTRUCTION_BYTES) != public):
        raise ValueError("original source or public instruction changed during preparation")
    fixture = {"schema": SCHEMA, "output": str(output), "repository": str(repository),
        "state": str(state), "captured_commit": captured_commit,
        "baseline_commit": prepared["manifest"]["payload"]["baseline_commit"],
        "source_provenance": {"path": str(source), "sha256": _sha(raw), "bytes": len(raw),
            "mode": source_mode, "selection": "caller-selected-public-pre-edit-input",
            "public_upstream": "https://github.com/bottlepy/bottle",
            "upstream_commit": "not_inferred_from_bytes"},
        "instruction_provenance": {"path": str(instruction), "sha256": _sha(public),
            "bytes": len(public), "selection": "caller-selected-public-task-instruction"},
        "prepared": prepared, "initial_context": initial,
        "learner": initial["codebase_autoencoder"],
        "worker_projection": {"max_bytes": 32768,
            "required_raw_paths": [prep.INSTRUCTION, prep.SMOKE],
            "bottle_source_disposition": "complete_source_CAS_with_fetch_required",
            "all_source_inline_status": ("not_attempted_exceeds_known_worker_byte_bound"
                                          if len(raw) > 32768 else "not_attempted"),
            "all_source_inline_rejection_executed": False},
        "training_scope": "declared_python_worker_inputs_including_public_smoke",
        "instruction_used_as_training_label": False, "source_bytes_unchanged": True,
        "canonical_tasks_created": False, "planning_provider_invoked": False,
        "admission_performed": False, "workers_started": False, "provider_calls": 0,
        "seconds": time.monotonic() - started, **_AUTHORITY}
    reference = _write(output / "fixture.json", fixture)
    return {**fixture, "fixture_artifact": reference}


def extract_terminal_codebase_metadata_records(fixture: dict) -> dict[str, list[dict]]:
    """Replay source/checkpoint bindings and export complete existing inventories.

    Parse, scanner and protocol records remain descriptive/candidate evidence.
    Learned feature/latent rows come from the verified trained artifact bytes.
    This performs no optimizer steps, planning, admission or checker execution.
    """
    import duckdb
    from . import terminal_indexed_preparation as prep
    from .terminal_initial_context import load_initial_context
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _scan_scoped_sources
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    from ipfs_datasets_py.logic.security_ir.doctor_header_contracts import (
        WsgiHeaderProtocolContract, analyze_http_header_contracts,
    )

    if type(fixture) is not dict or fixture.get("schema") != SCHEMA:
        raise ValueError("exact terminal codebase fixture required")
    output, repository, state = (Path(fixture[key]) for key in ("output", "repository", "state"))
    reference = fixture["fixture_artifact"]
    raw_fixture = _read(output / "fixture.json", MAX_ARTIFACT_BYTES)
    if (_sha(raw_fixture) != reference["sha256"]
            or reference["path"] != str(output / "fixture.json")
            or json.loads(raw_fixture) != {key: value for key, value in fixture.items()
                                           if key != "fixture_artifact"}):
        raise ValueError("terminal fixture declaration changed")
    loaded = load_initial_context(state=state, prepared=fixture["prepared"], require_empty_owner=True)
    descriptor, semantic = loaded["descriptor"], loaded["semantic"]
    sources = {name: _read(repository / name, MAX_SOURCE_BYTES)
               for name in fixture["prepared"]["worker_inputs"]}
    if any(_sha(raw) != semantic["manifest"][name]["sha256"] for name, raw in sources.items()):
        raise ValueError("terminal source differs from initial semantic scope")
    scanner = _scan_scoped_sources(sources, repository_id=descriptor["repository_id"], max_symbols=1024)
    provider = IpfsDatasetsSemanticStateProvider()
    rebuilt = provider.view_semantic_state_bundle(provider.build_semantic_state(scanner))
    if rebuilt.root.root_cid != descriptor["semantic"]["semantic_root_cid"]:
        raise ValueError("complete scanner export differs from initial semantic root")
    vectors = repository / ".runtime/terminal-vectors"
    evidence = json.loads(_read(vectors / "evidence.json", MAX_ARTIFACT_BYTES))
    evidence_sources = {"bottle.py": sources["bottle.py"].decode("utf-8")}
    replayed_evidence = build_program_evidence_index(evidence_sources)
    if _json_bytes(evidence) != _json_bytes(replayed_evidence.to_dict()):
        raise ValueError("native AST sidecar differs from exact captured source replay")
    if _json_bytes(json.loads(_read(vectors / "ast.json", MAX_ARTIFACT_BYTES))) != _json_bytes(replayed_evidence.ast_index.to_dict()):
        raise ValueError("native AST index differs from exact captured source replay")
    expected_source_hashes = {name: _sha(sources[name]) for name in evidence_sources}
    if (loaded["indexed"]["source_sha256"] != expected_source_hashes
            or descriptor["learned_embeddings"] is not False
            or loaded["indexed"].get("schema") != "native-lexical-vector-qualification@1"
            or loaded["indexed"].get("complete_permitted_scope") is not True):
        raise ValueError("retrieval declaration differs from expected lexical source scope")
    with duckdb.connect(str(vectors / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
        row = connection.execute("SELECT payload FROM snapshots WHERE id=?", [loaded["indexed"]["index_id"]]).fetchone()
    if row is None:
        raise ValueError("complete persisted retrieval snapshot is missing")
    snapshot_wire = json.loads(row[0])
    snapshot = CodeVectorIndexSnapshot.from_dict(snapshot_wire)
    expected_snapshot = _replay_lexical_snapshot(evidence=replayed_evidence, source_hashes=expected_source_hashes)
    if (snapshot.index_id != loaded["indexed"]["index_id"]
            or _json_bytes(snapshot_wire) != _json_bytes(expected_snapshot.to_dict())):
        raise ValueError("persisted retrieval snapshot differs from source/model/config replay")
    learner = descriptor["codebase_autoencoder"]
    learning = Path(learner["output"])
    features_raw, index_raw, checkpoint_raw = (_read(learning / (name + ".json"), MAX_ARTIFACT_BYTES)
                                              for name in ("features", "index", "checkpoint"))
    receipt = json.loads(_read(learning / "receipt.json", MAX_ARTIFACT_BYTES))
    if any(_sha(raw) != receipt[key] for raw, key in (
            (features_raw, "features_sha256"), (index_raw, "index_sha256"),
            (checkpoint_raw, "checkpoint_sha256"))):
        raise ValueError("trained artifact differs from bound training receipt")
    features, learned_index, checkpoint = map(json.loads, (features_raw, index_raw, checkpoint_raw))
    if ({row["row_id"] for row in features["rows"]}
            != {row["row_id"] for row in learned_index["ranks"]}
            or len(features["rows"]) != learner["sample_count"]
            or len(snapshot.rows) != loaded["indexed"]["symbols"]):
        raise ValueError("complete feature or vector inventory differs")
    protocol = WsgiHeaderProtocolContract(review_ref="experiment:declared-WSGI-start-response-protocol")
    header = analyze_http_header_contracts(sources["bottle.py"].decode(), protocol=protocol).to_dict()
    facts = [dict(fact, path=result["path"], source_sha256=result["source_sha256"])
             for result in evidence["results"] for fact in result["facts"]]
    records = {
        "sources": [{"path": name, "source_sha256": _sha(raw), "bytes": len(raw),
            "source_cid": semantic["manifest"][name]["source_cid"], "source_text": raw.decode(),
            "training_selected": name in learner["source_hashes"]}
            for name, raw in sorted(sources.items())],
        "ast": facts,
        "ast_indexes": [evidence],
        "symbols": [row.to_dict() for row in scanner.symbols],
        "artifacts": [row.to_dict() for row in scanner.artifacts],
        "kg": [row.to_dict() for row in scanner.edges],
        "contracts": [{"kind": "declared-public-task-contract", "task": fixture["prepared"]["spec"], **_AUTHORITY},
            {"kind": "source-bound-header-contract-analysis", "protocol_review_scope": "explicit_experiment_assumption",
             "analysis": header, "candidate_applied": False, **_AUTHORITY}],
        "features": features["rows"],
        "feature_frontiers": features["unsupported"],
        "vectors": [{**row, "checkpoint_sha256": learner["checkpoint_sha256"],
            "authority": "unverified_candidate_only", **_AUTHORITY} for row in learned_index["ranks"]],
        "retrieval_vectors": [row.to_dict() for row in snapshot.rows],
        "training": [{"learner": learner, "receipt": receipt, "checkpoint": checkpoint,
            "catalog": descriptor["codebase_autoencoder_catalog"], **_AUTHORITY}],
        "planning": [{"signed_manifest": fixture["prepared"]["manifest"],
            "declarations": fixture["prepared"]["spec"], "planning_inputs": fixture["prepared"]["manifest"]["payload"]["planning_inputs"],
            "initial_context": fixture["initial_context"], "descriptor": descriptor,
            "empty_world_capture": loaded["capture"], "worker_projection": fixture["worker_projection"], **_AUTHORITY}],
        "semantic_indexes": [scanner.to_dict()],
    }
    # A complete export is durable alongside its producer fixture, independently
    # byte-bound for the experiment's downstream native catalog/lake projection.
    bounded = bound_terminal_codebase_metadata_records(records)
    if _json_bytes(reconstruct_terminal_codebase_metadata_records(bounded)) != _json_bytes(records):
        raise ValueError("bounded metadata export lost producer fields")
    target = output / "metadata-records.json"
    if target.exists() or target.is_symlink():
        if _read(target, MAX_ARTIFACT_BYTES) != _json_bytes(bounded):
            raise ValueError("existing metadata export differs from verified producer replay")
    else:
        _write(target, bounded)
    return bounded


__all__ = ["prepare_terminal_codebase_fixture", "extract_terminal_codebase_metadata_records",
           "bound_terminal_codebase_metadata_records", "reconstruct_terminal_codebase_metadata_records"]
