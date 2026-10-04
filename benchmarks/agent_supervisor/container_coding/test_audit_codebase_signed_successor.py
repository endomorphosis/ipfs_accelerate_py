"""Authored pure receiving controls; none of these fixtures is native execution.

Synthetic full300 descriptive records, real Ed25519 signatures, and retained
stdout-shaped ordinary files exercise independent receiving joins. One case
checks an actual closed failed04 public-plan preimage against the same pure
contract. No product module, native owner, SQL, Git, Docker, fitting, or
inference is used here. Stored planning bytes attest no worker execution.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("authored_pure_signed_successor_audit", HERE / "audit_codebase_signed_successor.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
FALSE = {name: False for name in sorted(audit.inventory.AUTHORITY)}


def envelope(value):
    return {"artifact_cid": audit.structured(value), "value": value}


def implementation():
    files = {"authored.pure.fixture": "0" * 64}
    return {"files": files, "sha256": audit.sha(audit.wire(files)),
        "scope": "listed_local_files_only_not_execution_attestation"}


def make_context():
    """Build typed descriptive history; synthetic page CIDs attest no worker."""
    names = sorted(["README.md", "calc.py", "check_type.py", "check_offset.py"]
        + ["bulk%03d.py" % number for number in range(296)])
    sides, sources = [], {}
    for generation in (1, 2):
        entries, members = {}, []
        for number, name in enumerate(names):
            raw = (b"def increment(n: int) -> int:\n    return n + 1\n" if generation == 1 else audit.BEFORE) if name == "calc.py" else (name + "\n").encode()
            raw_hex = name.encode().hex()
            entry = {"schema": "ipfs-datasets.software-contracts.semantic-snapshot-entry@3", "path": name,
                "raw_path_hex": raw_hex, "kind": "artifact" if name == "README.md" else "source", "size_bytes": len(raw),
                "source_cid": audit.inventory.cid(raw), "opaque_reason": None, "git_blob_oid": None,
                "acquisition": "captured", "disposition": "filesystem", "head_blob_oid": None, "index_blob_oids": {}}
            entry["entry_cid"] = audit.structured(entry)
            key = "raw:" + raw_hex
            member = {"source_key": key, "path": name, "raw_path_hex": raw_hex, "entry_cid": entry["entry_cid"],
                "source_cid": entry["source_cid"], "ast_cid": None if name == "README.md" else audit.structured({
                    "authored_pure_ast": number, "generation": generation if name == "calc.py" else 1}),
                "parse_status": "unindexed" if name == "README.md" else "ok", "source_size_bytes": len(raw), "opaque_reason": None}
            entries[key] = entry
            members.append(member)
            if generation == 2:
                sources[name] = {"sha256": audit.sha(raw), "executable": False}
        sides.append({"entries": entries, "members": members})
    heads, receipts = [], []
    for generation in (1, 2):
        snapshot = audit.structured({"authored_pure_snapshot": generation})
        manifest = audit.structured({"authored_pure_manifest": generation})
        previous = None if not heads else heads[-1]
        receipt = {"schema": "codebase-publication-receipt@1", "operation_id": "authored-pure-" + str(generation),
            "request_cid": audit.structured({"schema": "codebase-publication-request@1", "repository_id": "authored:pure",
                "manifest_cid": manifest, "expected_head": previous}), "previous_head": previous,
            "repository_id": "authored:pure", "generation": generation, "manifest_cid": manifest,
            "snapshot_cid": snapshot, "ast_revision_id": "rev:authored:pure:snapshot:" + snapshot}
        receipts.append(receipt)
        heads.append(audit.source.receipt_head(receipt))
    ledger = audit.source.delta_ledger(*sides)
    delta = {"schema": "codebase-inventory-source-delta@1", "previous_head": heads[0], "current_head": heads[1],
        "previous_publication_receipt": receipts[0], "current_publication_receipt": receipts[1],
        "previous_membership_cid": audit.structured(sides[0]["members"]), "current_membership_cid": audit.structured(sides[1]["members"]),
        "capture_policy": {"max_entries": 512, "max_file_bytes": 65536, "exclusions": [".git", ".runtime"]},
        "ledger": ledger, "coverage": audit.source.delta_coverage(ledger),
        "limits": {"max_inventory_entries": 1024, "max_union_entries": 2048, "max_file_bytes": 65536,
            "max_manifest_bytes": 4 * audit.MIB, "max_delta_bytes": 8 * audit.MIB}, "optimized": True,
        "implementation": implementation(), "authority": deepcopy(FALSE), "numerical_reuse": False, "model_advanced": False,
        "removal_scope": "absent_from_current_complete_capture", "physical_absence_verified": False}
    parent_artifact = {"sha256": audit.sha(b"authored-pure-parent-checkpoint"), "bytes": 32}
    child_artifact = {"sha256": audit.sha(b"authored-pure-child-checkpoint"), "bytes": 31}
    parent = {"version_id": "authored:pure:parent", "variant_id": "authored:pure:variant", "artifact": parent_artifact,
        "artifact_cid": audit.inventory.cid(b"authored-pure-parent-checkpoint"), "contract_sha256": "1" * 64,
        "state_sha256": "2" * 64, "feature_space_sha256": "3" * 64, "latent_width": 8, "feature_columns": 2,
        "projection_ids": ["authored.pure.projection@1"], "projection_widths": {"authored.pure.projection@1": 2},
        "ancestry": [{"version_id": "authored:pure:parent", "artifact": parent_artifact}]}
    child = {**deepcopy(parent), "version_id": "authored:pure:child", "artifact": child_artifact,
        "artifact_cid": audit.inventory.cid(b"authored-pure-child-checkpoint"), "state_sha256": "4" * 64,
        "ancestry": [{"version_id": "authored:pure:child", "artifact": child_artifact}, *deepcopy(parent["ancestry"])]}
    limits = {"max_inventory_entries": 512, "page_entries": 32, "max_pages": 1024, "max_inferred_rows": 1024,
        "max_file_bytes": 65536, "max_manifest_bytes": 4 * audit.MIB, "max_target_bytes": 4 * audit.MIB,
        "max_input_bytes": 32 * audit.MIB, "max_output_bytes": 16 * audit.MIB}
    root = {"schema": "codebase-inventory-resume-root@1", "head": heads[1], "head_cid": audit.structured(heads[1]),
        "members": sides[1]["members"], "membership_cid": delta["current_membership_cid"], "model": child,
        "limits": limits, "optimized": True, "implementation": implementation(), "authority": deepcopy(FALSE)}
    selection = {"schema": "codebase-inventory-successor-scan@1", "source_delta_cid": audit.structured(delta),
        **{field: delta[field] for field in ("previous_head", "current_head", "previous_membership_cid", "current_membership_cid")},
        "previous_training_record_cid": audit.structured({"authored_pure_training": "parent"}),
        "training_record_cid": audit.structured({"authored_pure_training": "child"}), "previous_model": parent, "model": child,
        "root_cid": audit.structured(root), "scan_limits": limits, "optimized": True, "implementation": implementation(),
        "authority": deepcopy(FALSE), "training_performed_here": False, "inference_performed_here": False,
        "numerical_reuse": False, "model_head_promoted": False}
    pages = [{"page_cid": audit.inventory.cid(("authored-pure-page-" + str(start)).encode()), "start": start,
        "end": min(start + 32, 300), "membership_cid": audit.structured(root["members"][start:min(start + 32, 300)]),
        "inferred_rows": min(32, 300 - start), "dispositions": {"inferred": min(32, 300 - start)}} for start in range(0, 300, 32)]
    complete = {"schema": "codebase-inventory-resume-completion@1", "root_cid": audit.structured(root),
        "head_cid": root["head_cid"], "membership_cid": root["membership_cid"], "model_artifact_cid": child["artifact_cid"],
        "pages": pages, "coverage": {"inventory_entries": 300, "pages": 10, "inferred_rows": 300, "dispositions": {"inferred": 300}},
        "authority": deepcopy(FALSE)}
    scan = {"schema": "codebase-resume-completion-advisory@1", "root_cid": audit.structured(root),
        "completion_cid": audit.structured(complete), **{key: root[key] for key in ("head", "head_cid", "members", "membership_cid", "model", "limits", "implementation")},
        "pages": pages, "coverage": complete["coverage"], "root_record": root, "completion_record": complete, "authority": deepcopy(FALSE)}
    context = {"schema": "supervisor-codebase-successor-context@1", "selection": envelope(selection), "source_delta": envelope(delta),
        "inventory": {"schema": "supervisor-codebase-inventory-context@1", "scan": scan, "evidence": None, "authority": deepcopy(FALSE)},
        "authority": deepcopy(FALSE)}
    return context, sources


def rehash_context(context):
    """Update declared identities after a semantic negative example."""
    delta = context["source_delta"]["value"]
    context["source_delta"]["artifact_cid"] = audit.structured(delta)
    selected = context["selection"]["value"]
    selected["source_delta_cid"] = context["source_delta"]["artifact_cid"]
    scan = context["inventory"]["scan"]
    scan["root_cid"] = audit.structured(scan["root_record"])
    selected["root_cid"] = scan["root_cid"]
    scan["completion_record"]["root_cid"] = scan["root_cid"]
    scan["completion_cid"] = audit.structured(scan["completion_record"])
    context["selection"]["artifact_cid"] = audit.structured(selected)


def signing():
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = b"\xed\x01" + key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    alphabet, number, encoded = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz", int.from_bytes(raw, "big"), ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = alphabet[remainder] + encoded
    did = "did:key:z" + encoded
    def sign(value):
        return {"payload": value, "binding": {"identity": did, "profile_id": "authored-pure-profile",
            "signature": base64.b64encode(key.sign(audit.wire(value))).decode()}}
    return did, sign


def make_public(context, sources):
    declaration = audit.verify_successor_context(context)
    scan, inv = context["inventory"]["scan"], context["inventory"]
    compact = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    compact["member_paths"] = [row["path"] for row in scan["members"]]
    inv_declaration = {"schema": "supervisor-codebase-inventory-declaration@1", "full_context_cid": audit.structured(inv),
        "scan": compact, "evidence": None, "authority": deepcopy(FALSE)}
    roots = {"request_cid": audit.structured({"schema": "authored-successor-format-work-request@1",
        "objective": "Retain exact integer output and offset two while formatting the return expression",
        "task_keys": ["SUCCESSOR-TYPE", "SUCCESSOR-FORMAT"]}), "program_root": audit.structured({"authored_pure_program": 1}),
        "scan_cid": audit.structured({"head": scan["head"], "completed_scan": scan["completion_cid"]})}
    graph, specs = audit.authored_graph(roots)
    did, sign = signing()
    manifest = {"schema": "supervisor-local-benchmark-manifest@6", "repository": "/results/native/repository",
        "repository_cid": audit.structured({"authored_pure_repository": 1}), "baseline_commit": "0" * 40,
        "profile_dir": "/results/native/private/profile", "lifecycle_dir": "/results/native/private/lifecycle",
        "profile_content_id": "sha256:" + audit.sha(audit.wire({"authored_pure_profile": 1})),
        "sources": sources, "tasks": specs,
        "policy": deepcopy(audit.LOCAL_POLICY), "planning_roots": roots, "created_outputs": [],
        "codebase_inventory_context": inv_declaration, "codebase_successor_context": declaration}
    signed_manifest = sign(manifest)
    graph_cid = audit.structured(audit.semantic(graph))
    pending = audit.authored_pending(graph)
    tree = audit.structured({"schema": "supervisor-local-source-tree@1", "sources": sources})
    plan_id = audit.structured({"authored_pure_plan": 1})
    validation = {"schema": "ipfs_accelerate_py/agent-supervisor/formal-plan-validation@1", "validator_version": 1,
        "status": "consistent", "outcome": "consistent", "plan_id": plan_id, "plan_check_only": True,
        "bounds": {"schema": "ipfs_accelerate_py/agent-supervisor/formal-plan-validation-bounds@1", "configured": {},
            "domain_sizes": {"actors": 2, "events": 6, "evidence_requirements": 4, "fluents": 3, "formulas": 5,
                "goals": 1, "norms": 2, "provider_evidence": 0, "subgoals": 0, "tasks": 2, "temporal_constraints": 1},
            "effective_trace_bound": 16, "plan_trace_bound": 16, "search_nodes_explored": 82, "truncated_dimensions": []},
        "checks_performed": [], "consistency_level": "bounded_consistent", "countermodel": None, "evidence": [], "findings": [],
        "formula_ids": sorted(audit.structured({"authored_pure_formula": number}) for number in range(5)), "assumptions": []}
    planning = {"schema": "supervisor-local-planning-receipt@4", "manifest_cid": audit.structured(signed_manifest),
        "graph_cid": graph_cid, "plan_id": plan_id, "source_tree_id": tree, "pending_requirements": pending,
        "pending_cid": audit.structured(pending), "declared_input_closure": audit.declared_closure(graph_cid, tree, manifest),
        "plan_evidence": validation, "owner_profile_id": "authored-pure-profile", "planning_permitted": True,
        "completion_authority": False, "code_proof_authority": False, "production_activation": False,
        "codebase_inventory_context_cid": audit.structured(inv), "administrator_task_cids": sorted(row["content_id"] for row in graph["tasks"]),
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "codebase_successor_context_cid": audit.structured(context), "successor_selection_cid": declaration["selection_cid"],
        "source_delta_cid": declaration["source_delta_cid"]}
    selected_task = next(task for task in graph["tasks"] if task["task_key"] == "SUCCESSOR-FORMAT")
    public = {"schema": "supervisor-public-instruction@4", "repository": manifest["repository"],
        "task_cid": selected_task["content_id"], "task_id": "SUCCESSOR-FORMAT", "manifest": signed_manifest,
        "manifest_cid": planning["manifest_cid"], "owner_identity": did, "owner_profile_id": "authored-pure-profile",
        "source_path": "README.md", "source_sha256": sources["README.md"]["sha256"], "source_bytes": len(b"README.md\n"),
        "completion_authority": False, "publication_authority": False, "scope_expansion_authority": False,
        "inventory_plan_admission": {"graph": graph, "receipt": sign(planning)},
        "codebase_inventory_context": inv, "codebase_successor_context": context}
    public["context_cid"] = audit.structured(public)
    return public


def projection(public):
    plan, inv = public["inventory_plan_admission"], public["codebase_inventory_context"]
    planning, manifest = plan["receipt"]["payload"], public["manifest"]["payload"]
    scan = inv["scan"]
    spec = next(row for row in manifest["tasks"] if row["task_key"] == "SUCCESSOR-FORMAT")
    task = next(row for row in plan["graph"]["tasks"] if row["task_key"] == "SUCCESSOR-FORMAT")
    advisory = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    advisory["selected_task_members"] = [member for member in scan["members"] if member["path"] in spec["scope_paths"]]
    return {"schema": "supervisor-codebase-successor-worker-context@1", "task_cid": public["task_cid"], "task_id": "SUCCESSOR-FORMAT",
        "manifest_cid": public["manifest_cid"], "graph_cid": planning["graph_cid"], "planning_receipt_cid": audit.structured(plan["receipt"]),
        "codebase_inventory_context_cid": audit.structured(inv), "codebase_successor_context_cid": audit.structured(public["codebase_successor_context"]),
        "codebase_successor": manifest["codebase_successor_context"], "scan": advisory, "evidence": None,
        "administrator_task_cids": planning["administrator_task_cids"], "task_spec": spec,
        "dependency_task_cids": task["dependency_task_cids"], "pending_requirements": planning["pending_requirements"], "pending_cid": planning["pending_cid"],
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
        "publication_authority": False, "scope_expansion_authority": False, "authority": deepcopy(FALSE)}


def resign_public(public):
    """Keep every enclosing signed/CID preimage valid for semantic controls."""
    _, sign = signing()
    graph = public["inventory_plan_admission"]["graph"]
    old_tasks = {row["task_key"]: row["content_id"] for row in graph["tasks"]}
    goal = graph["goals"][0]
    for criterion in goal["acceptance"]:
        criterion["content_id"] = audit.structured(audit.semantic({key: value for key, value in criterion.items() if key != "content_id"}))
    goal["acceptance"].sort(key=lambda row: row["content_id"])
    goal["content_id"] = audit.structured(audit.semantic({key: value for key, value in goal.items() if key != "content_id"}))
    tasks = {row["task_key"]: row for row in graph["tasks"]}
    for key in ("SUCCESSOR-TYPE", "SUCCESSOR-FORMAT"):
        task = tasks[key]
        for field in ("outputs", "validations", "acceptance"):
            for row in task[field]:
                row["content_id"] = audit.structured(audit.semantic({name: value for name, value in row.items() if name != "content_id"}))
            task[field].sort(key=lambda row: row["content_id"])
        task["goal_cid"] = goal["content_id"]
        task["dependency_task_cids"] = [tasks["SUCCESSOR-TYPE"]["content_id"] if value == old_tasks["SUCCESSOR-TYPE"] else value
            for value in task["dependency_task_cids"]]
        task["content_id"] = audit.structured(audit.semantic({name: value for name, value in task.items() if name != "content_id"}))
    graph["tasks"].sort(key=lambda row: row["content_id"])
    public["task_cid"] = tasks["SUCCESSOR-FORMAT"]["content_id"]
    manifest = public["manifest"]["payload"]
    public["manifest"] = sign(manifest)
    public["manifest_cid"] = audit.structured(public["manifest"])
    plan = public["inventory_plan_admission"]
    planning = plan["receipt"]["payload"]
    planning["manifest_cid"] = public["manifest_cid"]
    planning["graph_cid"] = audit.structured(audit.semantic(graph))
    planning["administrator_task_cids"] = sorted(task["content_id"] for task in graph["tasks"])
    planning["pending_cid"] = audit.structured(planning["pending_requirements"])
    planning["source_tree_id"] = audit.structured({"schema": "supervisor-local-source-tree@1", "sources": manifest["sources"]})
    planning["declared_input_closure"] = audit.declared_closure(planning["graph_cid"], planning["source_tree_id"], manifest)
    plan["receipt"] = sign(planning)
    public["context_cid"] = audit.structured({key: value for key, value in public.items() if key != "context_cid"})


def retain(output, public):
    raw = audit.wire(public)
    path = output / "public.json"
    path.write_bytes(raw)
    path.chmod(0o444)
    successor = public["manifest"]["payload"]["codebase_successor_context"]
    staged_destination = "/authored-pure/staged/setup-seed"
    pin = {"bytes": 1, "sha256": "0" * 64}
    seed = {"schema": "source-successor-dispatch-staged-setup@1", "qualified": True,
        "source_namespace": "/authored-pure/closed/source", "staged_destination": staged_destination,
        "source_archive_inventory_cid": audit.structured({"authored_pure_archive": 1}),
        "audit": deepcopy(pin), "reader_controls": deepcopy(pin), "reader": deepcopy(pin), "native_result": deepcopy(pin),
        "copied_members": [], "copied_files": 0, "copied_bytes": 0, "reader_control_scope": "authored pure historical fixture",
        "reader_control_source_namespace": "/authored-pure/closed/source", "selected_producers": [],
        "current_head": successor["current_head"], "previous_head": successor["previous_head"],
        "root_cid": successor["root_cid"], "completion_cid": successor["completion_cid"], "selection_cid": successor["selection_cid"],
        "source_delta_cid": successor["source_delta_cid"], "selected_version_id": successor["model"]["version_id"],
        "previous_version_id": successor["previous_model"]["version_id"], "checkpoint_states": {},
        "inherited_setup_epochs": 2, "inherited_scan_pages": 10, "inherited_reference_pages": 1,
        "new_fitting_epochs": 0, "new_scan_pages": 0, "native_owners_opened": False,
        "fresh_native_receiving_required": True, "proof_authority": False, "source_execution_attested": False,
        "scan_execution_attested": False}
    (output / "source-dispatch-seed.json").write_bytes(audit.wire(seed))
    plan, scan = public["inventory_plan_admission"], public["codebase_inventory_context"]["scan"]
    planning = plan["receipt"]["payload"]
    prepared = {"artifact": str(path), "sha256": audit.sha(raw), "task_cid": public["task_cid"],
        "context_cid": public["context_cid"], "manifest_cid": public["manifest_cid"], "source_path": "README.md",
        "source_sha256": public["source_sha256"], "source_bytes": public["source_bytes"], "completion_authority": False,
        "scope_expansion_authority": False, "inventory_context_cid": audit.structured(projection(public)),
        "codebase_inventory_context_cid": audit.structured(public["codebase_inventory_context"]),
        "codebase_successor_context_cid": audit.structured(public["codebase_successor_context"]),
        "successor_selection_cid": successor["selection_cid"], "source_delta_cid": successor["source_delta_cid"]}
    boundary = {"schema": "supervisor-container-worker-boundary@1", "owner_uid": 1000, "worker_uid": 1001,
        "single_worker": True, "container_id": "5" * 64, "image_id": "sha256:" + "6" * 64,
        "namespaces": {name: name + ":[123]" for name in ("mnt", "net", "pid")},
        "allowed_worktree_roots": ["/opt/ipfs-supervisor/worktrees"], "validation_repository_roots": [public["repository"]],
        "owner_private_paths": ["/opt/ipfs-supervisor/state", "/results/native/private", staged_destination]}
    boundary_raw = audit.wire(boundary)
    (output / "container-boundary.json").write_bytes(boundary_raw)
    observed = {key: boundary[key] for key in ("schema", "container_id", "image_id", "namespaces", "owner_uid", "worker_uid")}
    observed.update(purpose="coding", manifest_sha256=audit.sha(boundary_raw), completion_authority=False)
    boundary_line = {"schema": "successor-authored-worker-boundary@1", "boundary": observed,
        "boundary_artifact": "/opt/ipfs-supervisor/container-boundary.json", "boundary_sha256": audit.sha(boundary_raw),
        "uid": 1001, "euid": 1001, "pid": 101, "gid": 1001, "groups": [1001], "provider_calls": 0, "training_steps": 0,
        "workspace": "/opt/ipfs-supervisor/worktrees/authored-pure", "private_access": {path: {
            "read": False, "write": False, "execute": False} for path in boundary["owner_private_paths"]}}
    inventory = {"context_cid": prepared["inventory_context_cid"], "codebase_inventory_context_cid": prepared["codebase_inventory_context_cid"],
        "root_cid": scan["root_cid"], "completion_cid": scan["completion_cid"], "membership_cid": scan["membership_cid"],
        "planning_receipt_cid": audit.structured(plan["receipt"]), "administrator_task_cids": planning["administrator_task_cids"],
        "pending_cid": planning["pending_cid"], "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False, "authority": deepcopy(FALSE)}
    successor_inclusion = {"context_cid": prepared["codebase_successor_context_cid"],
        **{key: successor[key] for key in ("selection_cid", "source_delta_cid", "previous_head", "current_head", "root_cid", "completion_cid")},
        "native_inventory_current_verified_here": False, "native_successor_current_verified_here": False, "authority": deepcopy(FALSE)}
    inclusion = {"schema": "supervisor-public-instruction-inclusion@4", "artifact": str(path), "artifact_sha256": audit.sha(raw),
        "context_cid": prepared["context_cid"], "task_cid": public["task_cid"], "task_id": "SUCCESSOR-FORMAT",
        "manifest_cid": prepared["manifest_cid"], "source_path": "README.md", "source_sha256": prepared["source_sha256"],
        "source_bytes": prepared["source_bytes"], "block_sha256": audit.sha(b"authored pure block"), "block_bytes": 19,
        "manifest_signature_verified": True, "verbatim_utf8": True, "semantic_minification_applied": False,
        "source_freshness_verified": True, "historical_replay": False, "completion_authority": False,
        "scope_expansion_authority": False, "extra_provider_calls": 0, "codebase_inventory": inventory, "codebase_successor": successor_inclusion}
    worker = {"schema": "source-successor-authored-native-worker@1", "status": "materialized", "pid": 101, "uid": 1001,
        "task_cid": public["task_cid"], "path": "calc.py", "before_sha256": audit.sha(audit.BEFORE), "after_sha256": audit.sha(audit.AFTER),
        "public_instruction": inclusion, "provider_calls": 0, "training_steps": 0, "proof_authority": False,
        "completion_authority": False, "native_completion_recorded_here": False}
    logs = output / "private/launch/authored-pure/implementation-logs"
    logs.mkdir(parents=True)
    log = logs / "authored-pure.log"
    log.write_bytes(audit.wire(boundary_line) + b"\n" + audit.wire(worker) + b"\n")
    return prepared, log, boundary_line, worker


class AuthoredPureReceivingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context, cls.sources = make_context()
        cls.public = make_public(deepcopy(cls.context), deepcopy(cls.sources))

    def context_control(self, change):
        value = deepcopy(self.context)
        change(value)
        rehash_context(value)
        with self.assertRaises(ValueError):
            audit.verify_successor_context(value)

    def receipt_control(self, change):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, log, boundary, worker = retain(output, deepcopy(self.public))
            change(prepared, boundary, worker)
            log.write_bytes(audit.wire(boundary) + b"\n" + audit.wire(worker) + b"\n")
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def signed_plan_control(self, change):
        value = deepcopy(self.public)
        change(value)
        resign_public(value)
        # Both retained signatures and every enclosing byte identity agree.
        audit.signature(value["manifest"])
        audit.signature(value["inventory_plan_admission"]["receipt"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, _, _, _ = retain(output, value)
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=value["task_cid"])

    def staged_seed_control(self, change):
        """Rehash the root boundary and stdout; only the semantic join differs."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, log, observed, worker = retain(output, deepcopy(self.public))
            boundary = audit.parse((output / "container-boundary.json").read_bytes())
            seed = audit.parse((output / "source-dispatch-seed.json").read_bytes())
            change(boundary, seed, observed)
            (output / "source-dispatch-seed.json").write_bytes(audit.wire(seed))
            raw = audit.wire(boundary)
            (output / "container-boundary.json").write_bytes(raw)
            observed["boundary"] = {key: boundary[key] for key in ("schema", "container_id", "image_id", "namespaces", "owner_uid", "worker_uid")}
            observed["boundary"].update(purpose="coding", manifest_sha256=audit.sha(raw), completion_authority=False)
            observed["boundary_sha256"] = audit.sha(raw)
            observed["private_access"] = {path: {"read": False, "write": False, "execute": False} for path in boundary["owner_private_paths"]}
            log.write_bytes(audit.wire(observed) + b"\n" + audit.wire(worker) + b"\n")
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def test_authored_pure_receipt_with_real_public_signatures(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, _, _, _ = retain(output, deepcopy(self.public))
            result = audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])
            self.assertTrue(result["verified"])
            for flag in ("process_origin_attested", "completion_authority", "proof_authority", "native_registry_opened",
                    "native_inventory_freshness_verified_here", "native_formal_compilation_reperformed_here", "numerical_execution_reperformed_here"):
                self.assertIs(result[flag], False)
            self.assertEqual(result["retained_staged_seed_binding"]["staged_destination"], "/authored-pure/staged/setup-seed")
            self.assertFalse(result["retained_staged_seed_binding"]["declared_staged_directory_opened"])
            self.assertFalse(result["retained_staged_seed_binding"]["original_source_archive_opened"])
            self.assertFalse(result["retained_staged_seed_binding"]["transport_qualification_reperformed_here"])

    def test_public_plan_cannot_duplicate_the_top_level_signed_manifest(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"].update(
            manifest=deepcopy(value["manifest"])))

    def test_native_profile_content_identity_cannot_be_replaced_with_a_cid(self):
        self.signed_plan_control(lambda value: value["manifest"]["payload"].update(
            profile_content_id=audit.structured({"unreviewed_profile": 1})))

    def test_retained_failed04_native_public_plan_has_router_two_field_shape(self):
        # The archived run failed before worker qualification. Reading these
        # exact ordinary public bytes exercises transport shape and authored
        # planning preimages, not native currentness or successful execution.
        native = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/"
                      "signed-successor-worker-qualification-20261003-04/native")
        reader = audit.Reader(native, seconds=30)
        raw = reader.raw("public-worker-artifact.json", 16 * 1024**2)
        public = audit.parse(raw)
        admission_raw = reader.raw("admission.json", 16 * 1024**2)
        admission = audit.parse(admission_raw)
        self.assertEqual(set(public["inventory_plan_admission"]), {"graph", "receipt"})
        self.assertEqual(set(admission), {"manifest", "graph", "receipt"})
        self.assertTrue(audit.same(public["manifest"], admission["manifest"]))
        self.assertTrue(audit.same(public["inventory_plan_admission"],
            {name: admission[name] for name in ("graph", "receipt")}))
        self.assertEqual(public["context_cid"], audit.structured(
            {name: value for name, value in public.items() if name != "context_cid"}))
        manifest = audit.signature(public["manifest"])
        planning = audit.signature(public["inventory_plan_admission"]["receipt"])
        graph = public["inventory_plan_admission"]["graph"]
        task_cids = {task["task_key"]: audit.prompt_task_record_cid(task) for task in graph["tasks"]}
        audit.verify_authored_plan(public, manifest, graph, planning, task_cids)
        self.assertEqual(reader.raw("public-worker-artifact.json", 16 * 1024**2), raw)
        self.assertEqual(reader.raw("admission.json", 16 * 1024**2), admission_raw)

    def test_closed_successor_context_rejects_extra_fields(self):
        self.context_control(lambda value: value.update(extra=True))

    def test_boolean_authority_cannot_be_integer_zero(self):
        self.context_control(lambda value: value["authority"].update(proof_authority=0))

    def test_coordinator_cannot_claim_training(self):
        self.context_control(lambda value: value["selection"]["value"].update(training_performed_here=True))

    def test_delta_cannot_claim_physical_absence(self):
        self.context_control(lambda value: value["source_delta"]["value"].update(physical_absence_verified=True))

    def test_delta_classification_is_replayed_after_rehash(self):
        self.context_control(lambda value: value["source_delta"]["value"]["ledger"][0].update(classification="removed"))

    def test_delta_source_comparison_is_replayed_after_rehash(self):
        self.context_control(lambda value: value["source_delta"]["value"]["ledger"][0].update(source_bytes_comparison="different"))

    def test_delta_coverage_is_replayed_after_rehash(self):
        self.context_control(lambda value: value["source_delta"]["value"]["coverage"].update(current_entries=299))

    def test_selection_direct_parent_is_bound(self):
        self.context_control(lambda value: value["selection"]["value"]["model"]["ancestry"][1].update(version_id="other-parent"))

    def test_selection_frozen_feature_basis_is_bound(self):
        self.context_control(lambda value: value["selection"]["value"]["previous_model"].update(feature_space_sha256="9" * 64))

    def test_model_raw_cid_is_bound_to_sha(self):
        self.context_control(lambda value: value["selection"]["value"]["model"].update(artifact_cid=audit.inventory.cid(b"other")))

    def test_evidence_for_full300_cannot_be_implied(self):
        self.context_control(lambda value: value["inventory"].update(evidence={"proof_authority": False}))

    def test_complete_coverage_requires_all_ten_pages(self):
        def change(value):
            value["inventory"]["scan"]["pages"].pop()
        self.context_control(change)

    def test_page_membership_subset_is_independently_bound(self):
        self.context_control(lambda value: value["inventory"]["scan"]["pages"][0].update(membership_cid=audit.structured([])))

    def test_page_identity_requires_raw_codec(self):
        self.context_control(lambda value: value["inventory"]["scan"]["pages"][0].update(page_cid=audit.structured({"wrong_codec": 1})))

    def test_page_gap_is_refused_after_rehash(self):
        self.context_control(lambda value: value["inventory"]["scan"]["pages"][1].update(start=33))

    def test_page_counts_reject_boolean(self):
        self.context_control(lambda value: value["inventory"]["scan"]["pages"][0].update(inferred_rows=True))

    def test_worker_pid_must_join_boundary(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(pid=102))

    def test_worker_uid_must_be_worker_uid(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(uid=1000))

    def test_worker_must_keep_exact_plus_two_preimage(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(before_sha256=audit.sha(b"n+1")))

    def test_worker_must_keep_exact_format_edit(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(after_sha256=audit.sha(b"n+3")))

    def test_worker_provider_calls_must_be_exact_zero(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(provider_calls=False))

    def test_worker_training_is_refused(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(training_steps=1))

    def test_worker_cannot_claim_native_completion(self):
        self.receipt_control(lambda prepared, boundary, worker: worker.update(native_completion_recorded_here=True))

    def test_private_authority_access_must_be_denied(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary["private_access"]["/results/native/private"].update(read=True))

    def test_third_staged_seed_private_access_must_be_denied(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary["private_access"]["/authored-pure/staged/setup-seed"].update(execute=True))

    def test_all_three_private_access_observations_are_required(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary["private_access"].pop("/authored-pure/staged/setup-seed"))

    def test_rehashed_boundary_third_path_must_join_staged_locator(self):
        self.staged_seed_control(lambda boundary, seed, observed: boundary["owner_private_paths"].__setitem__(2, "/authored-pure/other/setup-seed"))

    def test_changed_staged_locator_must_join_root_boundary(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed.update(staged_destination="/authored-pure/other/setup-seed"))

    def test_even_joined_third_locator_must_end_in_setup_seed(self):
        def change(boundary, seed, observed):
            boundary["owner_private_paths"][2] = "/authored-pure/staged/foreign-seed"
            seed["staged_destination"] = boundary["owner_private_paths"][2]
        self.staged_seed_control(change)

    def test_even_joined_third_locator_must_be_canonical_absolute(self):
        def change(boundary, seed, observed):
            boundary["owner_private_paths"][2] = "/authored-pure/staged/../setup-seed"
            seed["staged_destination"] = boundary["owner_private_paths"][2]
        self.staged_seed_control(change)

    def test_stage_cannot_bind_another_successor_selection(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed.update(selection_cid=audit.structured({"other_selection": 1})))

    def test_stage_cannot_bind_another_complete_scan(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed.update(completion_cid=audit.structured({"other_complete_scan": 1})))

    def test_stage_cannot_change_previous_head_scalar_type(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed["previous_head"].update(generation=True))

    def test_stage_cannot_claim_owner_opening(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed.update(native_owners_opened=True))

    def test_stage_cannot_claim_additional_fitting(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed.update(new_fitting_epochs=1))

    def test_stage_exact_zero_new_pages_cannot_be_boolean_false(self):
        self.staged_seed_control(lambda boundary, seed, observed: seed.update(new_scan_pages=False))

    def test_missing_retained_stage_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, _, _, _ = retain(output, deepcopy(self.public))
            (output / "source-dispatch-seed.json").unlink()
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def test_retained_stage_receipt_alias_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, _, _, _ = retain(output, deepcopy(self.public))
            path = output / "source-dispatch-seed.json"
            original = output / "original-stage.json"
            path.rename(original)
            path.symlink_to(original)
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def test_original_image_identity_must_join(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary["boundary"].update(image_id="sha256:" + "9" * 64))

    def test_original_namespace_identity_must_join(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary["boundary"]["namespaces"].update(pid="pid:[456]"))

    def test_worker_workspace_cannot_escape_allocated_root(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary.update(workspace="/results/native/private"))

    def test_worker_projection_digest_is_bound(self):
        self.receipt_control(lambda prepared, boundary, worker: prepared.update(inventory_context_cid=audit.structured({"wrong_projection": 1})))

    def test_successor_inclusion_cannot_claim_currentness(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_successor"].update(native_successor_current_verified_here=True))

    def test_successor_inclusion_delta_identity_is_bound(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_successor"].update(source_delta_cid=audit.structured({"wrong_delta": 1})))

    def test_inventory_inclusion_cannot_remove_administrator_task(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_inventory"].update(administrator_task_cids=[self.public["task_cid"]]))

    def test_inclusion_cannot_drop_pending_checks(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_inventory"].update(pending_cid=audit.structured([])))

    def test_resigned_receipt_cannot_drop_task_test(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["pending_requirements"].pop())

    def test_resigned_receipt_cannot_change_pending_phase(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["pending_requirements"][0].update(phase="pre_execution"))

    def test_resigned_pending_required_cannot_be_integer_one(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["pending_requirements"][0].update(required=1))

    def test_resigned_receipt_cannot_relax_pending_assurance(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["pending_requirements"][0].update(minimum_code_assurance="none"))

    def test_resigned_receipt_cannot_change_pending_subject(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["pending_requirements"][0].update(subject_ids=[value["task_cid"]]))

    def test_resigned_receipt_cannot_remove_fallback_command(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["pending_requirements"][0].update(fallback_check_ids=[]))

    def test_resigned_manifest_cannot_expand_output_scope(self):
        self.signed_plan_control(lambda value: value["manifest"]["payload"]["tasks"][1]["outputs"][0].update(path="check_offset.py"))

    def test_resigned_manifest_cannot_relax_exit_codes(self):
        self.signed_plan_control(lambda value: value["manifest"]["payload"]["tasks"][1]["validations"][0].update(expected_exit_codes=[0, 1]))

    def test_resigned_manifest_exit_code_cannot_be_boolean_false(self):
        self.signed_plan_control(lambda value: value["manifest"]["payload"]["tasks"][1]["validations"][0].update(expected_exit_codes=[False]))

    def test_resigned_graph_cannot_drop_dependency(self):
        def change(value):
            value["manifest"]["payload"]["tasks"][1]["dependencies"] = []
            next(task for task in value["inventory_plan_admission"]["graph"]["tasks"] if task["task_key"] == "SUCCESSOR-FORMAT")["dependency_task_cids"] = []
        self.signed_plan_control(change)

    def test_resigned_graph_cannot_replace_public_check(self):
        def change(value):
            value["manifest"]["payload"]["tasks"][1]["validations"][0]["argv"] = ["python3", "-c", "pass"]
            next(task for task in value["inventory_plan_admission"]["graph"]["tasks"] if task["task_key"] == "SUCCESSOR-FORMAT")["validations"][0]["argv"] = ["python3", "-c", "pass"]
        self.signed_plan_control(change)

    def test_resigned_graph_cannot_change_administrator_goal(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["graph"]["goals"][0].update(objective="Return n plus three"))

    def test_resigned_manifest_cannot_promote_policy(self):
        self.signed_plan_control(lambda value: value["manifest"]["payload"]["policy"].update(production_activation=True))

    def test_resigned_manifest_policy_false_cannot_be_integer_zero(self):
        self.signed_plan_control(lambda value: value["manifest"]["payload"]["policy"].update(production_activation=0))

    def test_resigned_plan_cannot_claim_completion(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"].update(completion_authority=True))

    def test_resigned_plan_cannot_remove_native_administrator_task(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"].update(removed_task_cids=[value["task_cid"]]))

    def test_resigned_plan_cannot_claim_observed_facts(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"].update(current_facts=["passes"]))

    def test_resigned_plan_evidence_must_remain_plan_only(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["plan_evidence"].update(plan_check_only=False))

    def test_resigned_plan_evidence_cannot_drop_pending_domain(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["plan_evidence"]["bounds"]["domain_sizes"].update(evidence_requirements=3))

    def test_resigned_plan_domain_count_cannot_be_boolean(self):
        self.signed_plan_control(lambda value: value["inventory_plan_admission"]["receipt"]["payload"]["plan_evidence"]["bounds"]["domain_sizes"].update(goals=True))

    def test_successor_inclusion_authority_false_cannot_be_integer_zero(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_successor"]["authority"].update(proof_authority=0))

    def test_inventory_inclusion_preserved_true_cannot_be_integer_one(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_inventory"].update(runtime_requirements_preserved=1))

    def test_inventory_inclusion_current_false_cannot_be_integer_zero(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_inventory"].update(native_inventory_current_verified_here=0))

    def test_boundary_completion_false_cannot_be_integer_zero(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary["boundary"].update(completion_authority=0))

    def test_successor_previous_generation_cannot_be_boolean(self):
        self.receipt_control(lambda prepared, boundary, worker: worker["public_instruction"]["codebase_successor"]["previous_head"].update(generation=True))

    def test_invalid_manifest_signature_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            value = deepcopy(self.public)
            value["manifest"]["binding"]["signature"] = base64.b64encode(b"0" * 64).decode()
            value["context_cid"] = audit.structured({key: field for key, field in value.items() if key != "context_cid"})
            prepared, _, _, _ = retain(output, value)
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=value["task_cid"])

    def test_old_boundary_schema_cannot_cross_into_new_profile(self):
        self.receipt_control(lambda prepared, boundary, worker: boundary.update(schema="inventory-authored-worker-boundary@1"))

    def test_duplicate_actual_worker_stdout_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, log, boundary, worker = retain(output, deepcopy(self.public))
            log.write_bytes(audit.wire(boundary) + b"\n" + audit.wire(worker) + b"\n" + audit.wire(worker) + b"\n")
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def test_worker_stdout_requires_preceding_same_log_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, log, boundary, worker = retain(output, deepcopy(self.public))
            log.write_bytes(audit.wire(worker) + b"\n" + audit.wire(boundary) + b"\n")
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def test_original_artifact_must_remain_readonly(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, _, _, _ = retain(output, deepcopy(self.public))
            Path(prepared["artifact"]).chmod(0o644)
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])

    def test_original_artifact_alias_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepared, _, _, _ = retain(output, deepcopy(self.public))
            path = Path(prepared["artifact"])
            original = output / "original.json"
            path.rename(original)
            path.symlink_to(original)
            with self.assertRaises(ValueError):
                audit.verify_authored_worker_receipt(output, prepared=prepared, task_cid=self.public["task_cid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
