"""Probe actual Doctor component execution without inventing authority receipts."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def run(root: Path, output: Path) -> dict:
    from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
        diagnose_repository,
        DoctorSourceUnit,
        DoctorAuthorityRoots,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters import (
        adapt_diagnostic_roots_to_deterministic,
        adapt_diagnostic_finding_to_deterministic,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import (
        adapt_program_source,
        build_language_edge_program_graph,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician import (
        DeterministicDoctorTactician,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider import (
        inspect_datasets_embedding_capability,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
        ProgramWorldDatabase,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
        SupervisorMetaIndex,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    output.mkdir(parents=True, exist_ok=False)
    target = "ipfs_accelerate_py/agent_supervisor/analysis/doctor_contract_adapters.py"
    raw = (root / target).read_bytes()
    cid = content_identity({"path": target, "sha256": hashlib.sha256(raw).hexdigest()})
    summary = {
        "scope": "single-module component probe",
        "source_id": cid,
        "full_system_qualified": False,
    }

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, default=str) + "\n")

    roots = DoctorAuthorityRoots(
        repository_id=content_identity({"root": str(root)}),
        forest_id=cid,
        tree_id=cid,
        overlay_id=cid,
        file_root_id=cid,
        blob_root_id=cid,
        config_id=content_identity({"scope": [target]}),
        policy_id=content_identity({"mode": "bounded-component-probe"}),
    )
    snapshot = diagnose_repository(
        sources=[DoctorSourceUnit(path=target, source_bytes=raw, blob_identity=cid)],
        authority_roots=roots,
    )
    save("diagnostics.json", snapshot.to_dict())
    det_roots = adapt_diagnostic_roots_to_deterministic(snapshot.authority_roots)
    tactician = DeterministicDoctorTactician()
    plans = []
    for finding in snapshot.findings[:8]:
        typed = adapt_diagnostic_finding_to_deterministic(
            finding, roots=det_roots, snapshot_id=snapshot.snapshot_cid
        )
        plan = tactician.plan_finding(typed, current_roots=det_roots)
        plans.append(plan.to_dict())
    save("tactician.json", plans)
    summary["tactician_findings_attempted"] = len(plans)
    summary["tactician_total_findings"] = len(snapshot.findings)
    adapter = adapt_program_source(raw.decode(), path=target, blob_identity=cid)
    graph = build_language_edge_program_graph([adapter], forest_id=cid)
    save("graph.json", graph.to_dict())
    hits = [
        node
        for node in graph.nodes
        if "adapt_diagnostic_finding_to_deterministic" in node.qualified_name
    ]
    save(
        "graph-query.json",
        {
            "query": "adapt_diagnostic_finding_to_deterministic",
            "graph_id": graph.graph_id,
            "hits": [n.to_dict() for n in hits],
            "semantic_authority": False,
        },
    )
    summary["graph"] = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "query_hits": len(hits),
        "graph_id": graph.graph_id,
    }
    # This is capability inspection, not a substitute random-vector backend.
    capability = inspect_datasets_embedding_capability()
    save("vector-capability.json", capability.to_dict())
    summary["vector"] = {
        "qualified": False,
        "reason": "No reviewed local embedding model bound; default hash-vector fixture is not semantic retrieval.",
    }
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    persisted = world.persist(
        {
            "task_id": "doctor-symbol-contract",
            "operation": "repair_observation",
            "board": "qualification",
            "source_id": cid,
            "graph_id": graph.graph_id,
            "diagnostic_snapshot_id": snapshot.snapshot_cid,
            "tactician_receipt_ids": [p.get("receipt_id") for p in plans],
            "completion_authority": False,
            "proposal_only": True,
        }
    )
    save("world-persist.json", persisted)
    retrieved = world.records_for_decision(task_id="doctor-symbol-contract")
    save("world-retrieval.json", retrieved)
    if retrieved["n"] != 1:
        raise RuntimeError("world record did not survive reopen")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, path, ref in [
        ("knowledge_graph", output / "graph.json", graph.graph_id),
        ("world_model", output / "world.duckdb", persisted["record_cid"]),
    ]:
        catalog = meta.register_catalog(
            kind=kind,
            locator_ref=str(path),
            repository_id=roots.repository_id,
            tree_id=cid,
            project=False,
        )
        meta.link_identity(
            subject_kind="path",
            subject_ref=target,
            catalog_id=catalog["catalog_id"],
            record_kind=kind,
            record_ref=ref,
            project=False,
        )
    projection = meta.project_ducklake()
    save("metadata-projection.json", projection)
    if projection["status"] != "projected":
        raise RuntimeError("metadata DuckLake projection unavailable")
    save(
        "metadata-retrieval.json", meta.compose_for_subject(subject_kind="path", subject_ref=target)
    )
    summary["world_model"] = {
        "persisted_and_reopened": True,
        "records": retrieved["n"],
        "scope": "repair observation memory, not a fully admitted supervisor world snapshot",
    }
    save("result.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root.resolve(), a.output.resolve())))
