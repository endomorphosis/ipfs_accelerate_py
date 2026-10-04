"""Capture live planning evidence without promoting memory to execution authority.

Intent records are read in one MVCC transaction. Semantic references are opened
through the datasets verifier. Missing owners stay unavailable: a plan, a source
scope, and stored claims alone cannot authorize scheduling or completion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import (
    IntentRepository,
    completion_evidence_projection_on_connection,
)
from .world_snapshot_builder import build_world_snapshot
from .world_snapshot_contracts import (
    COMPONENT_OWNERS,
    REQUIRED_COMPONENTS,
    mutable_snapshot,
    parse_world_snapshot,
)

MAX_CAPTURE_BYTES = 8_000_000


class IntentWorldSnapshotError(ValueError):
    """The live owners could not supply coherent bounded planning evidence."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _bounded(value: Any) -> dict:
    raw = _bytes(value)
    if len(raw) > MAX_CAPTURE_BYTES:
        raise IntentWorldSnapshotError("world evidence exceeds byte bound")
    return json.loads(raw)


def _world_cid(value: Any) -> str:
    # WorldSnapshot@1 has a closed sha256: vocabulary; producer CIDs remain
    # unchanged inside this explicitly content-addressed reference envelope.
    return "sha256:" + hashlib.sha256(_bytes(value)).hexdigest()


def capture_intent_world_snapshot(
    intent: IntentRepository,
    *,
    repository_id: str,
    task_cids: Sequence[str] = (),
    semantic_root_cid: str = "",
    get_semantic_block: Callable[[str], bytes] | None = None,
    coordinator: Any = None,
    coordination_epoch: int = 0,
    fencing_epoch: int = 0,
    transaction_owned_by_caller: bool = False,
) -> dict[str, Any]:
    """Join live plan/task evidence and verified semantic roots for planning.

    The repository identity is an explicit caller binding. This adapter does
    not claim an accepted repository tree from a scoped producer scan. The
    optional native coordinator supplies logical claim state, not a lease
    freshness check. Execution still requires its native fencing API.

    A bound IntentRepository requires an explicit existing caller transaction;
    this function never commits or rolls back that caller's transaction.
    """
    if not isinstance(intent, IntentRepository):
        raise TypeError("intent must be an IntentRepository")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise IntentWorldSnapshotError("repository identity is required")
    if intent.uses_bound_connection and not transaction_owned_by_caller:
        raise IntentWorldSnapshotError("bound intent requires caller transaction ownership")
    if transaction_owned_by_caller and not intent.uses_bound_connection:
        raise IntentWorldSnapshotError("caller transaction requires a bound intent connection")
    if bool(semantic_root_cid) != (get_semantic_block is not None):
        raise IntentWorldSnapshotError("semantic root and block reader are required together")
    if coordinator is not None:
        from ..merge.database_coordination import (
            DatabaseCoordinator,
            ProcessSerializedDatabaseCoordinator,
        )

        if not isinstance(coordinator, (DatabaseCoordinator, ProcessSerializedDatabaseCoordinator)):
            raise TypeError("coordinator must be a native database coordinator")
        coordination = _bounded(coordinator.coordination_registry_projection())
    else:
        coordination = None

    with intent._connection(write=False) as connection:
        owns = not transaction_owned_by_caller
        if owns:
            connection.execute("BEGIN TRANSACTION")
        try:
            reader = IntentRepository(bound_connection=connection, owner_id=intent.owner_id)
            plan = _plain(reader.plan_projection(task_cids=task_cids))
            completion = _plain(
                completion_evidence_projection_on_connection(
                    connection,
                    task_cids=task_cids,
                    transaction_owned_by_caller=True,
                )
            )
            heads = [
                head.to_dict()
                for goal in plan["goals"]
                if (head := reader.get_plan_head(goal["goal_cid"])) is not None
            ]
            watermark = reader.event_watermark()
            if watermark != completion["event_watermark"]:
                raise IntentWorldSnapshotError("intent event cursor changed within capture")
            if owns:
                connection.execute("COMMIT")
        except BaseException:
            if owns:
                connection.execute("ROLLBACK")
            raise

    if coordinator is not None and coordination != _bounded(
        coordinator.coordination_registry_projection()
    ):
        raise IntentWorldSnapshotError("coordination authority changed during capture")

    components = {
        name: {"status": "unavailable", "cid": "", "owner": COMPONENT_OWNERS[name]}
        for name in REQUIRED_COMPONENTS
    }
    payloads: dict[str, dict] = {}

    def bind(name: str, material: Any, *, status: str = "current") -> None:
        payload = {
            "schema": "supervisor-world-component-evidence@1",
            "component": name,
            "repository_id": repository_id,
            "material": _plain(material),
        }
        cid = _world_cid(payload)
        payloads[cid] = payload
        components[name] = {
            "status": status,
            "cid": cid,
            "evidence_cid": cid,
            "owner": COMPONENT_OWNERS[name],
            "repository_id": repository_id,
        }

    bind("repository", {"repository_id": repository_id, "binding": "explicit-caller-identity"})
    bind("objectives", plan["objectives"])
    bind("goal_population", [goal for goal in plan["goals"] if not goal["parent_goal_cid"]])
    bind(
        "subgoal_population",
        {
            "goals": [goal for goal in plan["goals"] if goal["parent_goal_cid"]],
            "edges": plan["goal_edges"],
        },
    )
    bind(
        "task_population",
        {
            "task_cids": sorted(task_cids),
            "tasks": plan["tasks"],
            "scope": "selected-tasks" if task_cids else "all-intent-tasks",
        },
    )
    if heads:
        bind("accepted_plan_root", {"active_heads": heads, "plans": plan["plans"]})
        bind("plan_revision", heads)
        components["task_population"]["plan_cid"] = components["accepted_plan_root"]["cid"]
    bind(
        "event_cursor",
        {"global_sequence": watermark, "plan_projection_cid": plan["projection_cid"]},
    )
    # Empty receipts are a real observed completion population, never a task-complete verdict.
    bind("completion_root", completion)

    if coordination is not None:
        selected = {task["task_cid"] for task in plan["tasks"]}
        registered = {task["task_cid"] for task in coordination["tasks"]}
        if not selected <= registered:
            raise IntentWorldSnapshotError("intent tasks absent from coordination authority")
        bind(
            "claims",
            {
                "task_cids": sorted(selected),
                "task_claims": [
                    claim for claim in coordination["task_claims"] if claim["task_cid"] in selected
                ],
                "lease_freshness_verified": False,
                "coordination_projection_root": coordination["projection_root"],
            },
        )

    producer_root = None
    if semantic_root_cid:
        from .datasets_adapter import IpfsDatasetsSemanticStateProvider
        from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
            verify_block_bytes,
        )

        provider = IpfsDatasetsSemanticStateProvider()
        consumed: dict[str, bytes] = {}

        def verified_block(cid: str) -> bytes:
            raw = get_semantic_block(cid)
            verify_block_bytes(cid, raw)
            consumed[cid] = raw
            if sum(map(len, consumed.values())) > MAX_CAPTURE_BYTES:
                raise IntentWorldSnapshotError("semantic evidence exceeds byte bound")
            return raw

        view = provider.open_verified_view(semantic_root_cid, verified_block)
        producer_root = view.root.to_dict()
        if producer_root["repository_id"] != repository_id:
            raise IntentWorldSnapshotError("semantic repository identity mismatch")
        bind(
            "datasets_repository",
            {"repository_id": producer_root["repository_id"], "root_cid": semantic_root_cid},
        )
        bind("datasets_semantic_state_root", producer_root)
        for name, field in (
            ("symbol_root", "symbol_node_index_cid"),
            ("capsule_index", "capsule_index_cid"),
            ("environment_bindings", "environment_binding_set_cid"),
        ):
            cid = producer_root[field]
            verified_block(cid)
            bind(name, {"root_cid": semantic_root_cid, "index_cid": cid})

    # Shared agreement commits to the actual joined records, not a fabricated
    # generation. Independent source identities remain in the component blocks.
    agreement = content_identity({name: record["cid"] for name, record in components.items()})
    for record in components.values():
        record["agreement"] = agreement
    admission = build_world_snapshot(
        components,
        repository_id=repository_id,
        coordination_epoch=coordination_epoch,
        fencing_epoch=fencing_epoch,
    )
    snapshot = mutable_snapshot(admission["snapshot"])
    planning_context = {
        "schema": "supervisor-intent-world-planning-context@1",
        "world_snapshot_cid": snapshot["snapshot_cid"],
        "repository_id": repository_id,
        "plan_projection_cid": plan["projection_cid"],
        "event_watermark": watermark,
        "objectives": plan["objectives"],
        "goals": plan["goals"],
        "goal_edges": plan["goal_edges"],
        "active_plan_heads": heads,
        "tasks": plan["tasks"],
        "semantic_root_cid": semantic_root_cid,
        "component_status": dict(admission["component_status"]),
        "unavailable_components": [
            name for name, record in components.items() if record["status"] == "unavailable"
        ],
        "schedulable": admission["schedulable"],
        "execution_authority": False,
        "completion_authority": False,
    }
    material = _bounded(
        {
            "schema": "supervisor-intent-world-capture@1",
            "snapshot": snapshot,
            "plan_projection": plan,
            "completion_projection": completion,
            "component_payloads": payloads,
            "planning_context": planning_context,
            "schedulable": admission["schedulable"],
            "unschedulable_reasons": admission["unschedulable_reasons"],
            "execution_authority": False,
            "completion_authority": False,
        }
    )
    return {**material, "capture_cid": content_identity(material)}


def persist_intent_world_snapshot(
    capture: Mapping[str, Any], *, output: Path, task_id: str
) -> dict:
    """Persist exact captured evidence in native world memory and DuckLake catalogs."""
    from .program_world_database import ProgramWorldDatabase
    from ..runtime.supervisor_meta_index import SupervisorMetaIndex

    material = _bounded(capture)
    claimed = material.pop("capture_cid", "")
    if claimed != content_identity(material):
        raise IntentWorldSnapshotError("capture content identity mismatch")
    task_ids = {
        value
        for task in material["plan_projection"]["tasks"]
        for value in (task["task_cid"], task["task_alias"])
    }
    if task_id not in task_ids:
        raise IntentWorldSnapshotError("world record must bind a captured task")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "intent-world.json"
    artifact.write_bytes(_bytes({**material, "capture_cid": claimed}))
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    record = world.persist(
        {
            "task_id": task_id,
            "board": "intent-world",
            "operation": "planning_snapshot",
            "capture_cid": claimed,
            "world_snapshot_cid": material["snapshot"]["snapshot_cid"],
            "planning_context": material["planning_context"],
            "proposal_only": True,
            "completion_authority": False,
        }
    )
    hydrated = world.records_for_decision(task_id=task_id, operation="planning_snapshot")
    if hydrated["n"] != 1 or hydrated["records"][0]["payload"]["capture_cid"] != claimed:
        raise IntentWorldSnapshotError("world snapshot hydration failed")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, locator, record_kind, ref in (
        (
            "world_model",
            output / "world.duckdb",
            "world_snapshot",
            material["snapshot"]["snapshot_cid"],
        ),
        ("taskboard", artifact, "plan_projection", material["plan_projection"]["projection_cid"]),
    ):
        catalog = meta.register_catalog(
            kind=kind,
            locator_ref=str(locator),
            repository_id=material["snapshot"]["repository_id"],
            project=False,
        )
        meta.link_identity(
            subject_kind="record_cid",
            subject_ref=claimed,
            catalog_id=catalog["catalog_id"],
            record_kind=record_kind,
            record_ref=ref,
            project=False,
        )
    projected = meta.project_ducklake()
    return {
        "capture_cid": claimed,
        "world_record": record,
        "metadata": projected,
        "artifact": str(artifact),
        "planning_context": material["planning_context"],
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "execution_authority": False,
        "completion_authority": False,
    }


def load_intent_world_context(
    *,
    artifact: Path,
    expected_sha256: str,
    task_id: str,
    repository_id: str,
    intent: IntentRepository | None = None,
) -> dict:
    """Load a task-bound planning context, optionally checking its live owner.

    The caller supplies a trusted artifact digest and repository identity and
    is responsible for admitting the artifact's filesystem location. Without
    the live IntentRepository, freshness is explicitly unchecked. Neither
    mode returns scheduling, fencing, or completion authority.
    """
    artifact = Path(artifact)
    if artifact.is_symlink() or artifact.stat().st_size > MAX_CAPTURE_BYTES:
        raise IntentWorldSnapshotError("world artifact is a symlink or exceeds byte bound")
    raw = artifact.read_bytes()
    if len(raw) > MAX_CAPTURE_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise IntentWorldSnapshotError("world artifact digest mismatch")
    capture = json.loads(raw)
    claimed = capture.pop("capture_cid", "")
    if capture.get("schema") != "supervisor-intent-world-capture@1" or claimed != content_identity(
        capture
    ):
        raise IntentWorldSnapshotError("world capture schema or content identity mismatch")
    snapshot = parse_world_snapshot(capture["snapshot"])
    if not repository_id or snapshot["repository_id"] != repository_id:
        raise IntentWorldSnapshotError("world repository identity mismatch")
    context = capture["planning_context"]
    plan = capture["plan_projection"]
    expected_status = {name: snapshot["components"][name]["status"] for name in REQUIRED_COMPONENTS}
    expected_unavailable = [
        name for name in REQUIRED_COMPONENTS if expected_status[name] == "unavailable"
    ]
    if (
        context.get("schema") != "supervisor-intent-world-planning-context@1"
        or context.get("repository_id") != repository_id
        or context.get("world_snapshot_cid") != snapshot["snapshot_cid"]
        or context.get("plan_projection_cid") != plan["projection_cid"]
        or any(
            context.get(field) != plan[field]
            for field in ("objectives", "goals", "goal_edges", "tasks")
        )
        or context.get("execution_authority") is not False
        or context.get("completion_authority") is not False
        or capture.get("execution_authority") is not False
        or capture.get("completion_authority") is not False
        or context.get("component_status") != expected_status
        or context.get("unavailable_components") != expected_unavailable
        or context.get("schedulable")
        is not all(status == "current" for status in expected_status.values())
        or context.get("event_watermark") != capture["completion_projection"]["event_watermark"]
        or plan["projection_cid"]
        != content_identity({key: value for key, value in plan.items() if key != "projection_cid"})
    ):
        raise IntentWorldSnapshotError("world planning context binding mismatch")
    task_ids = {value for task in plan["tasks"] for value in (task["task_cid"], task["task_alias"])}
    if task_id not in task_ids:
        raise IntentWorldSnapshotError("world task identity mismatch")
    for cid, payload in capture["component_payloads"].items():
        if _world_cid(payload) != cid:
            raise IntentWorldSnapshotError("world component content identity mismatch")
    for name in REQUIRED_COMPONENTS:
        component = snapshot["components"][name]
        if component["status"] == "unavailable":
            continue
        payload = capture["component_payloads"].get(component["cid"], {})
        if payload.get("component") != name or payload.get("repository_id") != repository_id:
            raise IntentWorldSnapshotError("world component reference binding mismatch")
    if intent is not None:
        scope = capture["component_payloads"][snapshot["components"]["task_population"]["cid"]][
            "material"
        ]
        live = intent.plan_projection(task_cids=scope["task_cids"])
        if live["projection_cid"] != plan["projection_cid"]:
            raise IntentWorldSnapshotError("world intent projection is stale")
        if intent.event_watermark() != context["event_watermark"]:
            raise IntentWorldSnapshotError("world intent event cursor is stale")
    return {**context, "intent_freshness_checked": intent is not None}


def minify_intent_world_worker_context(context: Mapping[str, Any]) -> dict:
    """Omit internal signed admission blobs from a verified advisory prompt.

    The source capture stays intact and digest-verified. This role projection
    retains exact content identities, task specifications and the captured
    roots; it does not pretend to be the full canonical plan projection.
    """
    result = _plain(context)
    omitted = []
    for task in result.get("tasks", []):
        body = task.get("body", {})
        if "local_planning_contract" in body:
            contract = body.pop("local_planning_contract")
            cid = content_identity(contract)
            body["local_planning_contract_cid"] = cid
            omitted.append({"task_cid": task["task_cid"], "contract_cid": cid,
                            "field": "body.local_planning_contract"})
    if omitted:
        result["worker_projection"] = {
            "schema": "supervisor-intent-world-worker-projection@1",
            "source_context_cid": content_identity(_plain(context)),
            "omitted_internal_contracts": omitted,
            "full_canonical_projection": False,
            "completion_authority": False, "execution_authority": False,
        }
    return result


def generate_prompt_goal_graph_with_world(
    request: Any,
    scan: Any,
    *,
    intent: IntentRepository,
    artifact: Path,
    expected_sha256: str,
    task_id: str,
    repository_id: str,
    router: Any = None,
    capabilities: Mapping[str, Any] | None = None,
    config: Any = None,
    max_world_context_bytes: int = 12_000,
) -> Any:
    """Feed live, task-bound world evidence to the native prompt planner.

    Summary records retain exact identities and revisions. Full declarations
    remain bound by the artifact and specification CIDs. Native request size,
    text, and proposal admission limits remain in force; nothing is truncated
    to make a request fit. This wrapper does not admit the resulting plan.
    """
    from ..prompt.prompt_goal_planner import generate_prompt_goal_graph

    context = load_intent_world_context(
        artifact=artifact,
        expected_sha256=expected_sha256,
        task_id=task_id,
        repository_id=repository_id,
        intent=intent,
    )
    summary = {
        key: context[key]
        for key in (
            "world_snapshot_cid",
            "repository_id",
            "plan_projection_cid",
            "event_watermark",
            "semantic_root_cid",
            "unavailable_components",
            "schedulable",
            "execution_authority",
            "completion_authority",
            "intent_freshness_checked",
        )
    }
    summary["schema"] = "supervisor-world-planning-summary@1"
    summary["artifact_sha256"] = expected_sha256
    summary["tasks"] = [
        {
            key: task[key]
            for key in (
                "task_cid",
                "task_alias",
                "goal_cid",
                "plan_cid",
                "status",
                "revision",
                "spec_cid",
            )
        }
        for task in context["tasks"]
    ]
    summary["goals"] = [
        {key: goal[key] for key in ("goal_cid", "parent_goal_cid", "title", "status", "revision")}
        for goal in context["goals"]
    ]
    summary["objectives"] = [
        {key: objective[key] for key in ("objective_id", "title", "status", "revision")}
        for objective in context["objectives"]
    ]
    summary["active_plan_heads"] = context["active_plan_heads"]
    if (
        type(max_world_context_bytes) is not int
        or not 1 <= max_world_context_bytes <= MAX_CAPTURE_BYTES
    ):
        raise IntentWorldSnapshotError("invalid planning context byte bound")
    if len(_bytes(summary)) > max_world_context_bytes:
        raise IntentWorldSnapshotError("world planning summary exceeds byte bound")
    return generate_prompt_goal_graph(
        request,
        scan,
        router=router,
        capabilities=capabilities,
        constraint_summaries={"constraint_summaries": [summary]},
        config=config,
    )
