"""Real producer preparation and durable observational selection references.

These fixtures exercise native scanning, capsule freshness, exact repacking,
DuckDB and DuckLake. The sidecar is host metadata, never extra worker input or
permission to omit sources. Native producer, storage, and source validation run
unchanged; two tests inject observer faults to check exception handling.
"""

import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.context_selection_observation import (
    encode_context_selection_observation,
    verify_context_selection_observation,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_selection import (
    select_worker_capsules,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    SemanticContextStale,
    _scan_scoped_sources,
    _worker_payload,
    load_semantic_worker_context,
    prepare_semantic_context,
    resolve_semantic_worker_context,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    IpfsDatasetsSemanticStateProvider,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
    ProgramWorldDatabase,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

TASK = "SELECTION-PREPARATION-001"
OBJECTIVE = "Inspect actual addition and retain the complete public instruction."
CATALOG_COLUMNS = (
    "catalog_cid, kind, locator_ref, exclusive_owner, attach_permitted, "
    "tree_id, recorded_at"
)
LINK_COLUMNS = (
    "link_cid, subject_kind, subject_ref, catalog_id, record_kind, record_ref, "
    "freshness_mtime_ns, recorded_at"
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    sources = {
        "code.py": (
            "def inspect(value):\n    return value + 1\n\n"
            "def preserve(value):\n    return inspect(value)\n"
        ),
        "support.py": "def harness_only(value):\n    return value\n",
        "instruction.md": "Keep the public instruction and its Unicode: λ, café.\n",
    }
    for name, source in sources.items():
        (repository / name).write_text(source, encoding="utf-8")
    return repository


def _prepare(repository, *, name="initial", program=None, **policy):
    output = repository / ".semantic" / name
    result = prepare_semantic_context(
        repository=repository,
        paths=["code.py", "instruction.md", "support.py"],
        required_raw_paths=["instruction.md"],
        program_paths=program,
        objective=policy.pop("objective", OBJECTIVE),
        task_id=TASK,
        output=output,
        **policy,
    )
    return output, result


def _load_arguments(repository, output, digest):
    return {
        "repository": repository,
        "artifact": (output / "worker-context.json").relative_to(repository).as_posix(),
        "expected_sha256": digest,
        "task_id": TASK,
    }


def _native_inputs(repository, payload):
    sources = {name: (repository / name).read_bytes() for name in payload["manifest"]}
    program = payload.get("program_paths")
    program_sources = sources if program is None else {name: sources[name] for name in program}
    state = _scan_scoped_sources(
        program_sources,
        repository_id=cid_for_payload({"repository": str(repository)}),
        max_symbols=payload["preparation_bounds"]["max_symbols"],
    )
    provider = IpfsDatasetsSemanticStateProvider()
    bundle = provider.build_semantic_state(state)
    view = provider.view_semantic_state_bundle(bundle)
    return sources, state, provider, bundle, view


def _regenerate_worker(repository, payload, **policy_override):
    """Replay the unchanged worker assembler, without invoking the observer."""
    sources, state, provider, bundle, view = _native_inputs(repository, payload)
    base = {
        key: value
        for key, value in payload.items()
        if key not in {"capsules", "admissions", "raw_sources", "pack", "worker_projection"}
    }
    projection = payload.get("worker_projection", {})
    policy = {
        "worker_query": projection.get("query", ""),
        "worker_capsule_limit": projection.get("max_capsules", 8),
        "worker_max_bytes": projection.get("max_bytes", 32768),
        **policy_override,
    }
    return _worker_payload(
        base=base,
        sources=sources,
        manifest=payload["manifest"],
        required=tuple(payload["required_raw_paths"]),
        state=state,
        bundle=bundle,
        view=view,
        provider=provider,
        context_input_tokens=payload["preparation_bounds"]["context_input_tokens"],
        **policy,
    )


def _assert_durable_observation(repository, output, result=None):
    raw = (output / "worker-context.json").read_bytes()
    payload = json.loads(raw)
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    records = world.records_for_decision(task_id=TASK)
    assert records["n"] == 1  # The sidecar must not create a second world row.
    assert records["completion_authority"] is False
    assert records["decision_authority"] is False
    assert records["ducklake_authoritative"] is False
    row = records["records"][0]
    assert row["operation"] == "context_prepared"
    world_payload = row["payload"]
    reference = world_payload["context_selection_observation"]
    assert set(reference) == {"schema", "artifact", "cid", "sha256", "bytes", "role"}
    assert reference["schema"] == "supervisor-context-selection-observation-ref@1"
    assert reference["artifact"] == "context-selection.json"
    assert reference["role"] == "coding"
    observation_raw = (output / reference["artifact"]).read_bytes()
    assert reference["bytes"] == len(observation_raw)
    assert reference["sha256"] == _sha(observation_raw)
    assert reference["cid"] == cid_for_bytes(observation_raw)
    assert (output / "blocks" / reference["cid"]).read_bytes() == observation_raw
    observation = verify_context_selection_observation(
        raw=observation_raw,
        expected_sha256=reference["sha256"],
        payload=payload,
        payload_bytes=raw,
        role="coding",
    )
    assert encode_context_selection_observation(observation) == observation_raw
    assert observation["payload"] == {
        "schema": payload["schema"], "sha256": _sha(raw), "bytes": len(raw)
    }
    assert observation["task_id"] == TASK
    assert observation["objective_sha256"] == _sha(payload["objective"].encode())
    assert observation["bindings"]["scope_cid"] == payload["scope_cid"]
    assert observation["bindings"]["semantic_root_cid"] == payload["semantic_root_cid"]
    assert observation["bindings"]["pack_cid"] == payload["pack"]["pack_cid"]
    assert observation["selection"]["selected_capsule_count"] == len(payload["capsules"])
    assert len(observation["capsules"]) == len(payload["capsules"])
    for ordinal, (native, admission, observed) in enumerate(zip(
        payload["capsules"], payload["admissions"], observation["capsules"]
    )):
        assert observed["ordinal"] == ordinal
        for key in ("capsule_cid", "stable_symbol_id", "version_cid", "source_cid"):
            assert observed[key] == native[key]
        assert observed["source_path"] == native["source_slice_path"]
        assert observed["proof_obligation_refs"] == native["proof_obligation_refs"]
        assert observed["admission"]["decision"] == admission["admission"]
        assert observed["admission"]["raw_source_required"] is admission["ref"]["raw_source_required"]
    assert observation["modality_evidence"]["codebase"]["status"] == "structural_producer_observed"
    for modality in ("intent", "security", "ui_ux"):
        assert observation["modality_evidence"][modality] == {"status": "not_observed_in_this_payload"}
    assert observation["provider_token_usage"] is None
    assert observation["token_savings"] is None
    assert all(value is False for value in observation["authority"].values())
    assert payload["objective"].encode() not in observation_raw
    assert (repository / "instruction.md").read_bytes().strip() not in observation_raw
    assert "context_selection_observation" not in payload
    assert "context_selection_observation_seconds" not in payload
    assert "context_selection_observation_seconds" not in world_payload
    assert world_payload["completion_authority"] is False
    assert world_payload["admitted"] is False
    assert world_payload["cas_completed"] is False
    if result is not None:
        assert result["context_selection_observation"] == reference
        elapsed = result["context_selection_observation_seconds"]
        assert type(elapsed) is float and 0 <= elapsed < float("inf")
        assert result["worker_payload_sha256"] == _sha(raw)
        assert result["compact_bytes"] == len(raw)
        assert result["ducklake"]["status"] == "projected"

    with duckdb.connect(
        str(output / "metadata.duckdb"), read_only=True, config={"threads": 1}
    ) as primary:
        catalogs = primary.execute(
            "SELECT " + CATALOG_COLUMNS + " FROM catalogs ORDER BY catalog_cid"
        ).fetchall()
        links = primary.execute(
            "SELECT " + LINK_COLUMNS + " FROM identity_links ORDER BY link_cid"
        ).fetchall()
        metadata = primary.execute(
            "SELECT c.locator_ref, c.tree_id, l.subject_kind, l.subject_ref, "
            "l.record_kind, l.record_ref, c.completion_authority, l.completion_authority "
            "FROM catalogs c JOIN identity_links l ON l.catalog_id=c.catalog_id "
            "WHERE c.kind='metadata'"
        ).fetchall()
    assert len(catalogs) == len(links) == 3
    assert metadata == [(
        str(output / reference["artifact"]), payload["scope_cid"],
        "record_cid", payload["scope_cid"], "metadata", reference["cid"], False, False,
    )]
    assert all(Path(item[2]).name != "control.duckdb" for item in catalogs)
    with duckdb.connect(
        ":memory:", config={"threads": 1, "memory_limit": "256MB"}
    ) as lake:
        lake.execute("SET autoinstall_known_extensions=false")
        lake.execute("SET autoload_known_extensions=false")
        lake.execute("LOAD ducklake")
        for directory, alias in (("world-lake", "world_history"), ("metadata-lake", "meta_history")):
            catalog = str(output / directory / "metadata.ducklake").replace("'", "''")
            data = str(output / directory / "parquet").replace("'", "''")
            lake.execute(
                "ATTACH 'ducklake:" + catalog + "' AS " + alias
                + " (READ_ONLY, DATA_PATH '" + data + "')"
            )
        history = lake.execute(
            "SELECT record_cid, payload_json, completion_authority "
            "FROM world_history.program_world_records"
        ).fetchall()
        assert len(history) == 1
        assert history[0][0] == row["record_cid"]
        assert json.loads(history[0][1]) == world_payload
        assert history[0][2] is False
        assert lake.execute(
            "SELECT " + CATALOG_COLUMNS + " FROM meta_history.catalogs ORDER BY catalog_cid"
        ).fetchall() == catalogs
        assert lake.execute(
            "SELECT " + LINK_COLUMNS + " FROM meta_history.identity_links ORDER BY link_cid"
        ).fetchall() == links
        for table in ("catalogs", "identity_links", "capsule_bindings"):
            assert lake.execute(
                "SELECT count(*) FROM meta_history." + table
                + " WHERE completion_authority IS DISTINCT FROM FALSE"
            ).fetchone()[0] == 0
    return payload, observation, reference


@pytest.mark.parametrize("program", [None, ["code.py"], []], ids=["legacy", "explicit", "empty-program"])
def test_native_preparation_records_sidecar_without_changing_worker_wire(tmp_path, program):
    repository = _repository(tmp_path)
    output, result = _prepare(repository, program=program)
    payload, observation, _ = _assert_durable_observation(repository, output, result)
    raw = (output / "worker-context.json").read_bytes()
    assert payload["schema"] == (
        "supervisor-semantic-worker-context@1" if program is None
        else "supervisor-semantic-worker-context@2"
    )
    assert observation["selection"]["selector"] == "full_producer_population"
    assert observation["selection"]["omitted_capsule_count"] == 0
    assert observation["selection"]["exact_byte_budget_repacked"] is False
    assert _regenerate_worker(repository, payload)[1] == raw
    assert load_semantic_worker_context(
        **_load_arguments(repository, output, result["worker_payload_sha256"])
    ).encode() == raw
    source_rows = {item["path"]: item for item in observation["sources"]}
    assert set(source_rows) == set(payload["manifest"])
    for name, binding in payload["manifest"].items():
        assert source_rows[name]["sha256"] == binding["sha256"]
        assert source_rows[name]["source_cid"] == binding["source_cid"]
        assert source_rows[name]["inline_raw"] is (name in payload["raw_sources"])
        assert source_rows[name]["required_raw"] is (name in payload["required_raw_paths"])
        assert source_rows[name]["program_source"] == (None if program is None else name in program)
    if program == []:
        assert result["capsules"] == result["worker_capsules"] == 0
        assert observation["capsules"] == []
        assert payload["raw_sources"] == {
            name: (repository / name).read_text(encoding="utf-8") for name in payload["manifest"]
        }


def test_observation_uses_final_repack_when_native_selection_is_trimmed_to_zero(tmp_path):
    repository = _repository(tmp_path)
    query = "inspect " * 1000
    assert 0 < len(query.encode()) <= 8192
    objective = ("Inspect the complete public behavior. " * 50).strip()
    initial, _ = _prepare(
        repository, name="budget-calibration", program=["code.py"],
        worker_query=query, worker_capsule_limit=1, objective=objective,
    )
    calibration = json.loads((initial / "worker-context.json").read_bytes())
    empty = _regenerate_worker(repository, calibration, worker_capsule_limit=0)
    budget = len(empty[1]) + 128
    assert 8192 <= budget <= 32768
    sources, state, provider, bundle, view = _native_inputs(repository, calibration)
    preliminary, _, _ = select_worker_capsules(
        bundle=bundle, view=view, provider=provider, symbols=state.symbols,
        sources=sources, required=calibration["required_raw_paths"],
        worker_query=query, worker_capsule_limit=1, worker_max_bytes=budget,
    )
    assert len(preliminary) == 1  # Exercise actual final trimming, not a zero-cap request.
    output, result = _prepare(
        repository, name="final-budget", program=["code.py"],
        worker_query=query, worker_capsule_limit=1, worker_max_bytes=budget,
        objective=objective,
    )
    payload, observation, _ = _assert_durable_observation(repository, output, result)
    selection = observation["selection"]
    assert selection["max_capsules"] == 1
    assert selection["exact_byte_budget_repacked"] is True
    assert selection["full_capsule_count"] == result["capsules"] > 0
    assert selection["selected_capsule_count"] == result["worker_capsules"] == 0
    assert selection["omitted_capsule_count"] == result["capsules"]
    assert payload["capsules"] == payload["admissions"] == observation["capsules"] == []
    assert result["compact_bytes"] <= budget
    assert payload["raw_sources"]["instruction.md"] == (
        repository / "instruction.md"
    ).read_text(encoding="utf-8")
    source_rows = {item["path"]: item for item in observation["sources"]}
    assert source_rows["code.py"]["fetch_required"] is True
    assert source_rows["instruction.md"]["required_raw"] is True
    assert source_rows["instruction.md"]["inline_raw"] is True
    assert _regenerate_worker(repository, payload)[1] == (output / "worker-context.json").read_bytes()


@pytest.mark.parametrize("program,changed", [(None, "code.py"), (["code.py"], "support.py")])
def test_stale_sources_require_refresh_and_new_durable_observation(tmp_path, program, changed):
    repository = _repository(tmp_path)
    output, result = _prepare(repository, program=program, worker_query="inspect", worker_capsule_limit=1)
    before, old_observation, old_reference = _assert_durable_observation(repository, output, result)
    worker_before = (output / "worker-context.json").read_bytes()
    sidecar_before = (output / "context-selection.json").read_bytes()
    arguments = _load_arguments(repository, output, result["worker_payload_sha256"])
    (repository / changed).write_text(
        (repository / changed).read_text(encoding="utf-8") + "# changed captured bytes\n",
        encoding="utf-8",
    )
    with pytest.raises(SemanticContextStale):
        load_semantic_worker_context(**arguments)
    resolved = resolve_semantic_worker_context(
        **arguments, refresh_output=repository / ".semantic/refresh", attempt_id="selection:2"
    )
    assert resolved["refreshed"] is True
    refreshed_output = (repository / resolved["artifact"]).parent
    after, new_observation, new_reference = _assert_durable_observation(repository, refreshed_output)
    assert before["scope_cid"] != after["scope_cid"]
    assert old_reference["cid"] != new_reference["cid"]
    assert old_observation["payload"]["sha256"] == result["worker_payload_sha256"]
    assert new_observation["payload"]["sha256"] == resolved["sha256"]
    assert after["refresh_lineage"]["previous_payload_sha256"] == result["worker_payload_sha256"]
    assert set(after["refresh_lineage"]["source_delta"]) == {changed}
    if changed == "support.py":
        assert before["semantic_root_cid"] == after["semantic_root_cid"]
    assert (output / "worker-context.json").read_bytes() == worker_before
    assert (output / "context-selection.json").read_bytes() == sidecar_before
    assert (output / "blocks" / old_reference["cid"]).read_bytes() == sidecar_before
    assert _regenerate_worker(repository, after)[1] == resolved["text"].encode()
    assert load_semantic_worker_context(
        **_load_arguments(repository, refreshed_output, resolved["sha256"])
    ) == resolved["text"]


def test_host_sidecar_is_separately_verified_and_never_worker_dispatch_authority(tmp_path):
    repository = _repository(tmp_path)
    output, result = _prepare(repository, program=["code.py"])
    payload, _, reference = _assert_durable_observation(repository, output, result)
    raw = (output / "worker-context.json").read_bytes()
    sidecar = output / "context-selection.json"
    sidecar.write_bytes(b'{"completion_authority":true}')
    with pytest.raises(ValueError):
        verify_context_selection_observation(
            raw=sidecar.read_bytes(), expected_sha256=reference["sha256"],
            payload=payload, payload_bytes=raw, role="coding",
        )
    # The existing loader still checks its native worker/source bindings; it
    # neither trusts nor injects this host observation into the model prompt.
    assert load_semantic_worker_context(
        **_load_arguments(repository, output, result["worker_payload_sha256"])
    ).encode() == raw
    (repository / "code.py").write_text("def inspect(value):\n    return value - 1\n", encoding="utf-8")
    with pytest.raises(SemanticContextStale):
        load_semantic_worker_context(
            **_load_arguments(repository, output, result["worker_payload_sha256"])
        )


def test_observation_resource_bound_preserves_real_worker_and_native_gates(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        context_selection_observation as observer,
    )

    def bounded_observation(**_arguments):
        raise observer.ContextSelectionObservationLimit("authored telemetry resource bound")

    monkeypatch.setattr(observer, "build_context_selection_observation", bounded_observation)
    repository = _repository(tmp_path)
    output, result = _prepare(repository, program=["code.py"], worker_query="inspect")
    unavailable = {
        "schema": "supervisor-context-selection-observation-unavailable@1",
        "status": "unavailable",
        "reason": "observation_resource_bound",
        "role": "coding",
    }
    assert result["context_selection_observation"] == unavailable
    assert not (output / "context-selection.json").exists()
    raw = (output / "worker-context.json").read_bytes()
    payload = json.loads(raw)
    assert _regenerate_worker(repository, payload)[1] == raw
    assert "context_selection_observation" not in payload
    records = ProgramWorldDatabase(
        output / "world.duckdb", output / "world-lake"
    ).records_for_decision(task_id=TASK)
    assert records["n"] == 1
    assert records["records"][0]["payload"]["context_selection_observation"] == unavailable
    with duckdb.connect(
        str(output / "metadata.duckdb"), read_only=True, config={"threads": 1}
    ) as primary:
        assert primary.execute("SELECT count(*) FROM catalogs").fetchone()[0] == 2
        assert primary.execute("SELECT count(*) FROM catalogs WHERE kind='metadata'").fetchone()[0] == 0
        assert primary.execute("SELECT count(*) FROM identity_links").fetchone()[0] == 2
    arguments = _load_arguments(repository, output, result["worker_payload_sha256"])
    assert load_semantic_worker_context(**arguments).encode() == raw
    (repository / "code.py").write_text("def inspect(value):\n    return value - 1\n", encoding="utf-8")
    with pytest.raises(SemanticContextStale):
        load_semantic_worker_context(**arguments)


def test_observation_logical_mismatch_is_not_masked_as_resource_unavailability(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        context_selection_observation as observer,
    )

    def mismatched_observation(**_arguments):
        raise ValueError("authored observation binding mismatch")

    monkeypatch.setattr(observer, "build_context_selection_observation", mismatched_observation)
    repository = _repository(tmp_path)
    with pytest.raises(ValueError, match="observation binding mismatch"):
        _prepare(repository, program=["code.py"])
    assert not (repository / ".semantic/initial").exists()
