"""Descriptive native index of finite observations, never a proof authority.

Every query checks the live repository and immutable receipt again. Stored facts
remain historical labels; only the finite observer/matcher can construct current
facts after fresh execution. This does not extend FormalVerificationCache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "terminal-codebase-finite-evidence-index@1"
DDL = (
    "CREATE TABLE match_record (singleton INTEGER PRIMARY KEY CHECK(singleton=1), match_sha256 VARCHAR NOT NULL, payload_json VARCHAR NOT NULL)",
    "CREATE TABLE observation_rows (ordinal INTEGER PRIMARY KEY, input BIGINT UNIQUE NOT NULL, output HUGEINT NOT NULL, payload_json VARCHAR NOT NULL)",
    "CREATE TABLE fact_rows (ordinal INTEGER PRIMARY KEY, fact_id VARCHAR UNIQUE NOT NULL, predicate_id VARCHAR UNIQUE NOT NULL, payload_json VARCHAR NOT NULL)",
    "CREATE TABLE clause_rows (ordinal INTEGER PRIMARY KEY, payload_json VARCHAR NOT NULL)",
)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return "sha256:" + hashlib.sha256(wire(value)).hexdigest()


def _catalog(cx):
    return {
        "tables": cx.execute("SELECT table_name,table_type FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall(),
        "columns": cx.execute("SELECT table_name,column_name,ordinal_position,data_type,is_nullable,column_default FROM information_schema.columns WHERE table_schema='main' ORDER BY table_name,ordinal_position").fetchall(),
        "constraints": sorted(cx.execute("SELECT table_name,constraint_type,constraint_text,constraint_column_names FROM duckdb_constraints() WHERE schema_name='main'").fetchall(), key=repr),
        "indexes": sorted(cx.execute("SELECT table_name,index_name,expressions,sql FROM duckdb_indexes() WHERE schema_name='main'").fetchall(), key=repr),
    }


def _require_schema(cx):
    import duckdb
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "64MB"}) as reference:
        for statement in DDL:
            reference.execute(statement)
        if _catalog(cx) != _catalog(reference):
            raise ValueError("finite native schema or constraints differ")


def _pin(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("canonical finite index artifact required")
    if path.stat().st_nlink != 1 or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("independent bounded finite index artifact required")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _records(match):
    observation = match["observation"]
    result = {"observations": [], "facts": [], "clauses": []}
    for ordinal, row in enumerate(observation.get("observations", [])):
        result["observations"].append({"ordinal": ordinal, "input": row["input"],
            "output": row["output"], "payload": row})
    for ordinal, fact in enumerate(match["current_facts"]):
        result["facts"].append({"ordinal": ordinal, "fact_id": fact["fact_id"],
            "predicate_id": fact["predicate"]["predicate_id"], "payload": fact})
    for ordinal, clause in enumerate(match["clause_results"]):
        result["clauses"].append({"ordinal": ordinal, "payload": clause})
    return result


def persist_finite_evidence_index(*, match, output: Path):
    """Retain the complete result and typed query columns in a new native DB."""
    import duckdb
    match = json.loads(wire(match))
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.exists():
        raise ValueError("new canonical finite index directory required")
    if not output.parent.is_dir():
        raise ValueError("finite index parent must already exist")
    rows = _records(match)
    if len(wire(match)) > 4 * 1024 * 1024:
        raise ValueError("finite index complete result exceeds bound")
    output.mkdir(mode=0o700)
    with duckdb.connect(str(output / "finite.duckdb"), config={"threads": 1, "memory_limit": "64MB"}) as cx:
        for statement in DDL:
            cx.execute(statement)
        cx.execute("INSERT INTO match_record VALUES (1,?,?)", [digest(match), wire(match).decode()])
        for row in rows["observations"]:
            cx.execute("INSERT INTO observation_rows VALUES (?,?,?,?)", [row["ordinal"], row["input"], row["output"], wire(row["payload"]).decode()])
        for row in rows["facts"]:
            cx.execute("INSERT INTO fact_rows VALUES (?,?,?,?)", [row["ordinal"], row["fact_id"], row["predicate_id"], wire(row["payload"]).decode()])
        for row in rows["clauses"]:
            cx.execute("INSERT INTO clause_rows VALUES (?,?)", [row["ordinal"], wire(row["payload"]).decode()])
        _require_schema(cx)
    (output / "match.json").write_bytes(wire(match) + b"\n")
    result = {"schema": SCHEMA, "output": str(output), "match_sha256": digest(match),
        "head": match["observation"]["head"],
        "counts": {key: len(value) for key, value in rows.items()},
        "artifacts": [_pin(output / "finite.duckdb"), _pin(output / "match.json")],
        "cache_authority": "historical_candidate_only", "fresh_observation_required": True,
        "proof_authority": False, "execution_authority": False, "completion_authority": False}
    result["manifest_id"] = digest(result)
    (output / "manifest.json").write_bytes(wire(result) + b"\n")
    return result


def query_finite_evidence_index(*, index, repository, expected_head, expected,
        scheduler=None, parent_lease=None, cancel_event=None):
    """Verify every stored field and live root before returning historical rows."""
    import duckdb
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    if type(index) is not RepositoryCodebaseIndex:
        raise ValueError("native current codebase index required")
    expected = json.loads(wire(expected))
    fields = {"schema", "output", "match_sha256", "head", "counts", "artifacts",
        "cache_authority", "fresh_observation_required", "proof_authority",
        "execution_authority", "completion_authority", "manifest_id"}
    if (type(expected) is not dict or set(expected) != fields or expected["schema"] != SCHEMA
            or expected["cache_authority"] != "historical_candidate_only"
            or expected["fresh_observation_required"] is not True
            or any(expected[key] is not False for key in ("proof_authority", "execution_authority", "completion_authority"))):
        raise ValueError("finite index requires exact historical-only manifest fields")
    body = dict(expected)
    identifier = body.pop("manifest_id", None)
    if identifier != digest(body) or wire(expected["head"]) != wire(expected_head.to_dict()):
        raise ValueError("finite index root or manifest identity differs")
    output = Path(expected["output"])
    if not output.is_absolute() or output.resolve(strict=True) != output:
        raise ValueError("canonical finite index output required")
    if (type(expected["artifacts"]) is not list or
            [row["path"] for row in expected["artifacts"]] != [str(output / "finite.duckdb"), str(output / "match.json")]):
        raise ValueError("complete exact finite artifact population required")
    if json.loads((output / "manifest.json").read_bytes()) != expected:
        raise ValueError("finite index manifest differs")
    if any(_pin(pin["path"]) != pin for pin in expected["artifacts"]):
        raise ValueError("finite index artifact differs")
    def observe():
        return index.observe_current(repository, expected_head=expected_head,
            scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event)
    observe()
    match = json.loads((output / "match.json").read_bytes())
    rows = _records(match)
    if (type(expected["counts"]) is not dict or set(expected["counts"]) != set(rows)
            or any(type(value) is not int or value < 0 for value in expected["counts"].values())):
        raise ValueError("finite index requires exact nonnegative integer counts")
    if {key: len(value) for key, value in rows.items()} != expected["counts"]:
        raise ValueError("finite index complete counts differ")
    if digest(match) != expected["match_sha256"]:
        raise ValueError("finite index result identity differs")
    receipt = match["observation"]
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    if index.artifacts.get(cid_for_structured(receipt)) != receipt:
        raise ValueError("finite observation immutable receipt differs")
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import validate_finite_integer_observation
    validate_finite_integer_observation(receipt, expected_head=expected_head,
        contract=IntegerOffsetContract.from_dict(receipt["contract"]),
        inputs=receipt["domain_inputs"], tool_policy=receipt["tool_policy"])
    with duckdb.connect(str(output / "finite.duckdb"), read_only=True,
            config={"threads": 1, "memory_limit": "64MB"}) as cx:
        _require_schema(cx)
        if cx.execute("SELECT singleton,match_sha256,payload_json FROM match_record").fetchall() != [(1, digest(match), wire(match).decode())]:
            raise ValueError("complete finite native result differs")
        wanted = {"observation_rows": [(r["ordinal"], r["input"], r["output"], wire(r["payload"]).decode()) for r in rows["observations"]],
            "fact_rows": [(r["ordinal"], r["fact_id"], r["predicate_id"], wire(r["payload"]).decode()) for r in rows["facts"]],
            "clause_rows": [(r["ordinal"], wire(r["payload"]).decode()) for r in rows["clauses"]]}
        for table, values in wanted.items():
            if cx.execute("SELECT * FROM " + table + " ORDER BY ordinal").fetchall() != values:
                raise ValueError("finite typed native rows differ: " + table)
    observe()
    if any(_pin(pin["path"]) != pin for pin in expected["artifacts"]):
        raise ValueError("finite index changed during query")
    return {"schema": "terminal-codebase-finite-evidence-query@1", "manifest_id": identifier,
        "head": expected_head.to_dict(), "match_sha256": digest(match),
        "counts": expected["counts"], "live_source_checked_before_and_after": True,
        "current_facts": [], "historical_fact_ids": [r["fact_id"] for r in rows["facts"]],
        "cache_authority": "historical_candidate_only", "fresh_observation_required": True,
        "proof_authority": False, "execution_authority": False, "completion_authority": False}
