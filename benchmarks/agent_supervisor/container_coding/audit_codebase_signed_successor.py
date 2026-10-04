"""Independent bounded receipt receiving for a signed successor worker.

Only retained ordinary bytes and Ed25519 public keys are read. No product
module, native owner, SQL connection, Git command, model, or Docker daemon is
opened. Native execution and boundary receipts are historical observations;
this reader attests neither process origin nor currentness nor correctness.
"""
from __future__ import annotations

from collections import Counter
import base64
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re


def _reader(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("signed_successor_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _reader("audit_codebase_inventory_resume_worker")
source = _reader("audit_codebase_source_successor")
Reader = inventory.Reader
need = inventory.need
parse = inventory.parse
sha = inventory.sha
wire = inventory.wire
structured = inventory.structured
signature = inventory.signature
semantic = inventory.semantic
prompt_task_record_cid = inventory.prompt_task_record_cid
false_authority = inventory.false_authority
SignedSuccessorAuditError = inventory.InventoryWorkerAuditError
SCHEMA = "source-successor-authored-worker-receipt-verification@1"
BEFORE = b"def increment(n: int) -> int:\n    return n + 2\n"
AFTER = b"def increment(n: int) -> int:\n    return (2 + n)\n"
MIB = 1024 ** 2
LOCAL_POLICY = {"schema": "supervisor-isolated-benchmark-planning-policy@1",
    "scope": "signed-local-benchmark-only", "pending_assurance": "candidate",
    "explicit_proof_obligations": [], "external_ir_roots": [], "production_activation": False}
SCOPES = ["README.md", "calc.py", "check_offset.py", "check_type.py"]
SPEC_SCOPES = ["calc.py", "check_type.py", "check_offset.py", "README.md"]
STAGED_FIELDS = {"schema", "qualified", "source_namespace", "staged_destination", "source_archive_inventory_cid",
    "audit", "reader_controls", "reader", "native_result", "copied_members", "copied_files", "copied_bytes",
    "reader_control_scope", "reader_control_source_namespace", "selected_producers", "current_head", "previous_head",
    "root_cid", "completion_cid", "selection_cid", "source_delta_cid", "selected_version_id", "previous_version_id",
    "checkpoint_states", "inherited_setup_epochs", "inherited_scan_pages", "inherited_reference_pages",
    "new_fitting_epochs", "new_scan_pages", "native_owners_opened", "fresh_native_receiving_required",
    "proof_authority", "source_execution_attested", "scan_execution_attested"}


def closed(value, fields, name):
    need(type(value) is dict and set(value) == set(fields), "closed " + name + " required")


def same(left, right):
    """Canonical bytes preserve exact scalar types in descriptive joins."""
    return wire(left) == wire(right)


def integer(value, maximum, minimum=0):
    need(type(value) is int and minimum <= value <= maximum, "bounded exact integer required")


def digest(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "exact SHA-256 required")


def artifact(value):
    closed(value, {"sha256", "bytes"}, "model artifact")
    digest(value["sha256"])
    integer(value["bytes"], 16 * MIB, 1)


def model_shape(value):
    closed(value, {"version_id", "variant_id", "artifact", "artifact_cid", "contract_sha256",
        "state_sha256", "feature_space_sha256", "latent_width", "feature_columns", "projection_ids",
        "projection_widths", "ancestry"}, "model identity")
    for key in ("version_id", "variant_id"):
        source.text(value[key], 512)
    artifact(value["artifact"])
    source.check_cid(value["artifact_cid"], source=True)
    expected = "b" + base64.b32encode(b"\x01\x55\x12\x20" + bytes.fromhex(value["artifact"]["sha256"])).decode().lower().rstrip("=")
    need(expected == value["artifact_cid"], "model artifact CID/SHA binding differs")
    for key in ("contract_sha256", "state_sha256", "feature_space_sha256"):
        digest(value[key])
    integer(value["latent_width"], 64, 1)
    integer(value["feature_columns"], 1024, 1)
    ids = value["projection_ids"]
    need(type(ids) is list and 1 <= len(ids) <= 2 and all(type(key) is str for key in ids)
         and ids == sorted(set(ids)), "ordered projection identities required")
    closed(value["projection_widths"], ids, "projection layout")
    for width in value["projection_widths"].values():
        integer(width, 1024, 1)
    need(sum(value["projection_widths"].values()) == value["feature_columns"], "feature layout differs")
    ancestry = value["ancestry"]
    need(type(ancestry) is list and 1 <= len(ancestry) <= 8, "bounded model ancestry required")
    for row in ancestry:
        closed(row, {"version_id", "artifact"}, "ancestor identity")
        source.text(row["version_id"], 512)
        artifact(row["artifact"])
    need(len({row["version_id"] for row in ancestry}) == len(ancestry)
         and ancestry[0] == {"version_id": value["version_id"], "artifact": value["artifact"]},
         "model ancestry identity differs")


def member_shape(member):
    closed(member, {"source_key", "path", "raw_path_hex", "entry_cid", "source_cid", "ast_cid",
        "parse_status", "source_size_bytes", "opaque_reason"}, "inventory member")
    raw = member["raw_path_hex"]
    need(type(raw) is str and 0 < len(raw) <= 8192 and re.fullmatch(r"(?:[0-9a-f]{2})+", raw),
         "canonical raw path required")
    need(member["source_key"] == "raw:" + raw and member["path"] == source.raw_display(bytes.fromhex(raw)),
         "captured source path differs")
    source.check_cid(member["entry_cid"])
    if member["source_cid"] is not None:
        source.check_cid(member["source_cid"], source=True)
    if member["ast_cid"] is not None:
        source.check_cid(member["ast_cid"])
    if member["source_size_bytes"] is not None:
        integer(member["source_size_bytes"], 2**63 - 1)
    need(member["parse_status"] in {"ok", "partial", "failed", "opaque", "unindexed"}, "native parse status required")
    opaque = member["parse_status"] == "opaque"
    need(opaque == (member["opaque_reason"] is not None)
         and (member["ast_cid"] is None) == (member["parse_status"] in {"opaque", "unindexed"})
         and (opaque or (member["source_cid"] is not None and member["source_size_bytes"] is not None)),
         "source/AST opacity binding differs")
    if opaque:
        source.text(member["opaque_reason"], 512)


def verify_delta(envelope):
    """Replay typed historical delta joins from its full entry/member ledger."""
    closed(envelope, {"artifact_cid", "value"}, "source delta envelope")
    value = envelope["value"]
    closed(value, {"schema", "previous_head", "current_head", "previous_publication_receipt",
        "current_publication_receipt", "previous_membership_cid", "current_membership_cid", "capture_policy",
        "ledger", "coverage", "limits", "optimized", "implementation", "authority", "numerical_reuse",
        "model_advanced", "removal_scope", "physical_absence_verified"}, "source delta")
    need(value["schema"] == "codebase-inventory-source-delta@1" and type(value["optimized"]) is bool
         and len(wire(value)) <= 8 * MIB and envelope["artifact_cid"] == structured(value), "source delta identity differs")
    false_authority(value["authority"])
    need(value["numerical_reuse"] is value["model_advanced"] is value["physical_absence_verified"] is False
         and value["removal_scope"] == "absent_from_current_complete_capture", "source delta authority differs")
    previous = source.receipt_head(value["previous_publication_receipt"])
    current = source.receipt_head(value["current_publication_receipt"])
    need(previous == value["previous_head"] and current == value["current_head"]
         and value["current_publication_receipt"]["previous_head"] == previous
         and current["repository_id"] == previous["repository_id"], "source successor publication binding differs")
    caps = {"max_inventory_entries": 1024, "max_union_entries": 2048, "max_file_bytes": 65536,
        "max_manifest_bytes": 4 * MIB, "max_delta_bytes": 8 * MIB}
    closed(value["limits"], caps, "delta limits")
    for field, cap in caps.items():
        integer(value["limits"][field], cap, 1)
    need(len(wire(value)) <= value["limits"]["max_delta_bytes"], "declared delta byte limit exceeded")
    policy = value["capture_policy"]
    closed(policy, {"max_entries", "max_file_bytes", "exclusions"}, "complete capture policy")
    integer(policy["max_entries"], value["limits"]["max_inventory_entries"], 1)
    integer(policy["max_file_bytes"], value["limits"]["max_file_bytes"], 1)
    need(type(policy["exclusions"]) is list and all(type(name) is str and name for name in policy["exclusions"])
         and policy["exclusions"] == sorted(set(policy["exclusions"])), "ordered capture exclusions required")
    ledger = value["ledger"]
    need(type(ledger) is list and 0 < len(ledger) <= value["limits"]["max_union_entries"], "bounded complete delta ledger required")
    sides = [{"entries": {}, "members": []}, {"entries": {}, "members": []}]
    for row in ledger:
        closed(row, {"source_key", "classification", "previous", "current", "source_bytes_comparison",
            "ast_identity_comparison"}, "source delta row")
        need(row["previous"] is not None or row["current"] is not None, "empty source union row")
        for number, field in enumerate(("previous", "current")):
            observed = row[field]
            if observed is None:
                continue
            closed(observed, {"entry", "member"}, "source delta side")
            entry, member = observed["entry"], observed["member"]
            source.entry_shape(entry)
            member_shape(member)
            need(row["source_key"] == member["source_key"]
                 and all(member[left] == entry[right] for left, right in (("path", "path"), ("raw_path_hex", "raw_path_hex"),
                     ("entry_cid", "entry_cid"), ("source_cid", "source_cid"), ("source_size_bytes", "size_bytes"),
                     ("opaque_reason", "opaque_reason"))), "delta entry/member identity differs")
            need(row["source_key"] not in sides[number]["entries"], "duplicate source delta member")
            sides[number]["entries"][row["source_key"]] = entry
            sides[number]["members"].append(member)
    for number, field in enumerate(("previous_membership_cid", "current_membership_cid")):
        members = sides[number]["members"]
        need(len(members) <= policy["max_entries"] and value[field] == structured(members), "complete delta membership identity differs")
    need(same(ledger, source.delta_ledger(*sides)) and same(value["coverage"], source.delta_coverage(ledger)),
         "complete source union/classification/coverage differs")
    source.implementation_shape(value["implementation"])
    return value


def coverage(value, *, pages=None):
    closed(value, {"inventory_entries", "inferred_rows", "dispositions"} | ({"pages"} if pages is not None else set()), "scan coverage")
    integer(value["inventory_entries"], 1024)
    integer(value["inferred_rows"], value["inventory_entries"])
    counts = value["dispositions"]
    need(type(counts) is dict and set(counts) <= {"inferred", "opaque", "unindexed", "parse_failed", "parse_partial",
        "unsupported_target", "feature_incompatible", "deferred_budget"}, "closed scan dispositions required")
    for count in counts.values():
        integer(count, 1024, 1)
    need(sum(counts.values()) == value["inventory_entries"] and counts.get("inferred", 0) == value["inferred_rows"],
         "scan coverage conservation differs")
    if pages is not None:
        need(type(value["pages"]) is int and value["pages"] == pages, "complete scan page count differs")


def verify_successor_context(context):
    """Validate descriptive full records, without page execution or currentness."""
    try:
        closed(context, {"schema", "selection", "source_delta", "inventory", "authority"}, "successor context")
        need(context["schema"] == "supervisor-codebase-successor-context@1" and len(wire(context)) <= 12 * MIB,
             "bounded full successor context required")
        false_authority(context["authority"])
        delta = verify_delta(context["source_delta"])
        inv = context["inventory"]
        closed(inv, {"schema", "scan", "evidence", "authority"}, "inventory context")
        need(inv["schema"] == "supervisor-codebase-inventory-context@1" and inv["evidence"] is None,
             "successor300 context must retain evidence null")
        false_authority(inv["authority"])
        scan = inv["scan"]
        closed(scan, {"schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid", "members",
            "model", "pages", "coverage", "limits", "implementation", "root_record", "completion_record", "authority"}, "completion advisory refs")
        need(scan["schema"] == "codebase-resume-completion-advisory@1", "completion advisory schema differs")
        false_authority(scan["authority"])
        selected = context["selection"]
        closed(selected, {"artifact_cid", "value"}, "successor selection envelope")
        selection = selected["value"]
        model_shape(selection["previous_model"])
        model_shape(selection["model"])
        root_envelope = {"artifact_cid": scan["root_cid"], "value": scan["root_record"]}
        source.verify_selection(selected, context["source_delta"], root_envelope, selection["previous_model"])
        for field in ("previous_training_record_cid", "training_record_cid"):
            source.check_cid(selection[field])
        members = source.verify_root(root_envelope, context["source_delta"], selection["model"])
        need(selection["model"]["latent_width"] == 8 and len(members) == 300
             and [row["raw_path_hex"] for row in members] == sorted({row["raw_path_hex"] for row in members})
             and len({row["entry_cid"] for row in members}) == len(members), "complete ordered300 8D membership required")
        root = scan["root_record"]
        completion = scan["completion_record"]
        closed(completion, {"schema", "root_cid", "head_cid", "membership_cid", "model_artifact_cid", "pages", "coverage", "authority"}, "resume completion")
        need(completion["schema"] == "codebase-inventory-resume-completion@1"
             and scan["completion_cid"] == structured(completion)
             and all(completion[key] == expected for key, expected in (("root_cid", scan["root_cid"]),
                 ("head_cid", root["head_cid"]), ("membership_cid", root["membership_cid"]),
                 ("model_artifact_cid", root["model"]["artifact_cid"]))), "complete root/model binding differs")
        false_authority(completion["authority"])
        pages = completion["pages"]
        need(type(pages) is list and len(pages) == 10, "complete300 requires ten32 pages")
        counts, inferred, offset = Counter(), 0, 0
        for page in pages:
            closed(page, {"page_cid", "start", "end", "membership_cid", "inferred_rows", "dispositions"}, "page descriptor")
            source.check_cid(page["page_cid"], source=True)
            source.check_cid(page["membership_cid"])
            end = min(offset + 32, 300)
            need(type(page["start"]) is type(page["end"]) is int and page["start"] == offset and page["end"] == end
                 and page["membership_cid"] == structured(members[offset:end]), "complete page membership/order differs")
            coverage({"inventory_entries": end - offset, "inferred_rows": page["inferred_rows"], "dispositions": page["dispositions"]})
            counts.update(page["dispositions"])
            inferred += page["inferred_rows"]
            offset = end
        need(len({page["page_cid"] for page in pages}) == 10, "duplicate durable page identity")
        coverage(completion["coverage"], pages=10)
        need(completion["coverage"] == {"inventory_entries": 300, "pages": 10, "inferred_rows": inferred,
            "dispositions": dict(sorted(counts.items()))}, "complete scan coverage differs")
        expected = {"schema": "codebase-resume-completion-advisory@1", "root_cid": scan["root_cid"],
            "completion_cid": scan["completion_cid"], **{key: root[key] for key in ("head", "head_cid", "membership_cid", "members", "model", "limits", "implementation")},
            "pages": pages, "coverage": completion["coverage"], "root_record": root, "completion_record": completion,
            "authority": context["authority"]}
        need(same(scan, expected), "full inventory advisory records differ")
        return {"schema": "supervisor-codebase-successor-declaration@1", "full_context_cid": structured(context),
            "selection_cid": selected["artifact_cid"], "source_delta_cid": context["source_delta"]["artifact_cid"],
            **{key: selection[key] for key in ("previous_head", "current_head", "previous_membership_cid", "current_membership_cid", "previous_model", "model", "root_cid")},
            "completion_cid": scan["completion_cid"], "inventory_context_cid": structured(inv), "authority": context["authority"]}
    except source.SourceSuccessorAuditError as error:
        raise SignedSuccessorAuditError(str(error)) from error


def native_record(kind, **fields):
    """Reviewed PromptWorkflow v1 identity; embedded self-identities stay bound."""
    value = {"schema": "ipfs_accelerate_py/agent-supervisor/prompt-" + kind + "-record@1",
        "contract_version": 1, **fields}
    return {**value, "content_id": structured(semantic(value))}


def authored_graph(roots):
    """Independently authored two-task meaning, without native compiler imports."""
    policy = structured(LOCAL_POLICY)
    output = native_record("output", path="calc.py", effect="modify", media_type="text/x-python")
    criteria = [native_record("acceptance", criterion_key="inventory-" + kind,
        criterion="The public " + kind + " check passes", evidence_cids=[], validation_keys=["public-" + kind])
        for kind in ("type", "offset")]
    goal = native_record("goal", goal_key="SUCCESSOR-GOAL", parent_goal_cid="", dependency_goal_cids=[],
        title="Format increment", objective="Retain exact integer output and offset two while formatting the return expression",
        rationale="Independent administrator task declarations; inventory features are advisory", scope_paths=SCOPES,
        acceptance=sorted(criteria, key=lambda row: row["content_id"]), assumptions=[], evidence_cids=[],
        risks=[], provenance={}, status="proposed", created_at_ms=0, updated_at_ms=0)
    tasks, specs = [], []
    for kind, criterion in zip(("type", "offset"), criteria):
        check = native_record("validation", validation_key="public-" + kind,
            argv=["python3", "-B", "check_" + kind + ".py"], cwd=".", expected_exit_codes=[0], policy_cid=policy)
        task = native_record("task", task_key="SUCCESSOR-TYPE" if kind == "type" else "SUCCESSOR-FORMAT",
            goal_cid=goal["content_id"], dependency_task_cids=[] if not tasks else [tasks[0]["content_id"]],
            objective="Preserve the public type check" if kind == "type" else "Parenthesize the return expression while preserving the public offset check",
            rationale="Keep every original administrator task", scope_paths=SCOPES, outputs=[output], validations=[check],
            acceptance=[criterion], assumptions=[], evidence_cids=[], policy_roots=[policy], predicted_files=["calc.py"],
            bundle="", fallback_behavior="fail_closed", parallel_lane="", priority="P1", provenance={},
            resource_class="cpu-medium", risks=[], status="proposed", track="prompt-workflow", created_at_ms=0, updated_at_ms=0)
        tasks.append(task)
        specs.append({"task_key": task["task_key"], "scope_paths": SPEC_SCOPES,
            "dependencies": [] if kind == "type" else ["SUCCESSOR-TYPE"],
            "outputs": [{key: output[key] for key in ("path", "effect", "media_type")}],
            "validations": [{key: check[key] for key in ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}],
            "acceptance": [{key: criterion[key] for key in ("criterion_key", "criterion", "evidence_cids", "validation_keys")}]})
    graph = {"schema": "ipfs_accelerate_py/agent-supervisor/prompt-goal-graph@1", "contract_version": 1,
        **roots, "policy_roots": [policy], "goals": [goal], "tasks": sorted(tasks, key=lambda row: row["content_id"]),
        "evidence": [], "uncertainty_debt": [], "unresolved_questions": [], "status": "proposed", "created_at_ms": 0,
        "updated_at_ms": 0}
    return graph, specs


def authored_pending(graph):
    """Rebuild every native task test and goal review, including their IDs."""
    formal_policy = structured({"namespace": "prompt-formal-policy", "policy_roots": graph["policy_roots"]})
    pending = []
    def row(subject, criterion, ordinal, *, goal=False, validation=None):
        kind = "review" if goal else "test"
        preimage = {"id": criterion["criterion_key"], "kind": kind,
            "check_ids": criterion["validation_keys"], "source_scope_ids": SCOPES}
        requirement = structured({"goal_id" if goal else "task_id": subject, "criterion": preimage, "ordinal": ordinal})
        fallback = list(criterion["validation_keys"])
        if validation is not None:
            fallback.append(wire({key: validation[key] for key in
                ("argv", "cwd", "expected_exit_codes", "policy_cid", "validation_key")}).decode())
        pending.append({"schema": "ipfs_accelerate_py/agent-supervisor/formal-plan-evidence-requirement@1",
            "contract_version": 1, "requirement_id": requirement, "kind": kind, "subject_ids": [subject],
            "source_scope_ids": SCOPES if goal else ["path:calc.py"], "minimum_code_assurance": "candidate",
            "freshness_seconds": None, "fallback_check_ids": sorted(set(fallback)),
            "metadata": {"criterion": preimage, "policy_ids": [formal_policy]}, "phase": "post_execution", "required": True})
    for task in graph["tasks"]:
        row(task["content_id"], task["acceptance"][0], 0, validation=task["validations"][0])
    goal = graph["goals"][0]
    for ordinal, criterion in enumerate(goal["acceptance"]):
        row(goal["content_id"], criterion, ordinal, goal=True)
    return sorted(pending, key=lambda item: item["requirement_id"])


def declared_closure(graph_cid, tree, manifest):
    """Replay the complete native descriptive star closure over signed inputs."""
    decision = structured({"local_planning_graph": graph_cid, "tree": tree})
    common = {"root_id": tree, "source_root_id": tree, "provenance": "source", "trust": "verified",
        "authority": "descriptive_input", "version": "local-declared-inputs@1"}
    def named(namespace, value):
        return namespace + ":sha256:" + sha(wire(value, ascii=False))
    def node(kind, record, node_id):
        value = {"schema": "ipfs_accelerate_py/agent-supervisor/semantic-dependency-node@1",
            "node_id": node_id, "kind": kind, **common, "provenance_id": node_id, "record": record}
        return {**value, "content_id": named("semantic-node", value), "authoritative": True}
    nodes, edges = [node("decision", {"graph_cid": graph_cid}, decision)], []
    declarations = [("file", {"path": path, **source_record}) for path, source_record in manifest["sources"].items()]
    declarations.extend(("action", spec) for spec in manifest["tasks"])
    declarations.append(("authorization", {"profile_content_id": manifest["profile_content_id"], "policy": manifest["policy"]}))
    for kind, record in declarations:
        node_id = structured({"kind": kind, "declaration": record})
        nodes.append(node(kind, record, node_id))
        edge = {"schema": "ipfs_accelerate_py/agent-supervisor/semantic-dependency-edge@1", "source": decision,
            "target": node_id, "kind": "requires", **common, "provenance_id": structured(record), "mandatory": True, "record": {}}
        edges.append({**edge, "edge_id": named("semantic-edge", edge), "authoritative": True})
    nodes.sort(key=lambda item: item["node_id"])
    edges.sort(key=lambda item: item["edge_id"])
    graph = {"schema": "ipfs_accelerate_py/agent-supervisor/semantic-dependency-graph@1",
        "root_id": tree, "nodes": nodes, "edges": edges}
    graph.update(graph_id=named("semantic-graph", graph), node_count=len(nodes), edge_count=len(edges))
    paths = {item["node_id"]: [decision] if item["node_id"] == decision else [decision, item["node_id"]] for item in nodes}
    closure = {"schema": "ipfs_accelerate_py/agent-supervisor/mandatory-dependency-closure@1", "root_id": tree,
        "decision_id": decision, "node_ids": sorted(paths), "edge_ids": [item["edge_id"] for item in edges], "paths": paths}
    closure.update(closure_id=named("mandatory-closure", closure), annotation_node_ids=[], annotation_edge_ids=[],
        bounds={"max_nodes": 16384, "max_edges": 65536, "max_depth": 256, "max_annotations": 4096}, complete=True, truncated=False)
    return {"scope": "signed-declarations-and-complete-tracked-file-inventory", "semantic_dependency_graph": graph,
        "closure": closure, "code_semantic_closure_claimed": False, "proof_authority": False}


def verify_authored_plan(public, manifest, graph, planning, task_cids):
    closed(public, {"schema", "repository", "task_cid", "task_id", "manifest", "manifest_cid", "owner_identity",
        "owner_profile_id", "source_path", "source_sha256", "source_bytes", "completion_authority", "publication_authority",
        "scope_expansion_authority", "context_cid", "inventory_plan_admission", "codebase_inventory_context", "codebase_successor_context"}, "successor public artifact")
    # The public transport keeps its signed manifest at the artifact root.
    # Its plan contains the full graph and receipt, exactly as the router emits.
    closed(public["inventory_plan_admission"], {"graph", "receipt"}, "full public planning admission")
    closed(manifest, {"schema", "repository", "repository_cid", "baseline_commit", "profile_dir", "lifecycle_dir",
        "profile_content_id", "sources", "tasks", "policy", "planning_roots", "created_outputs",
        "codebase_inventory_context", "codebase_successor_context"}, "successor signed manifest")
    closed(planning, {"schema", "manifest_cid", "graph_cid", "plan_id", "source_tree_id", "pending_requirements", "pending_cid",
        "declared_input_closure", "plan_evidence", "owner_profile_id", "planning_permitted", "completion_authority",
        "code_proof_authority", "production_activation", "codebase_inventory_context_cid", "administrator_task_cids",
        "current_facts", "removed_task_cids", "runtime_requirements_preserved", "codebase_successor_context_cid",
        "successor_selection_cid", "source_delta_cid"}, "successor signed planning receipt")
    need(same(manifest["policy"], LOCAL_POLICY) and manifest["created_outputs"] == []
         and public["task_id"] == "SUCCESSOR-FORMAT" and manifest["repository"] == public["repository"]
         and all(public[key] is False for key in ("completion_authority", "publication_authority", "scope_expansion_authority"))
         and planning["planning_permitted"] is True
         and all(planning[key] is False for key in ("completion_authority", "code_proof_authority", "production_activation"))
         and planning["owner_profile_id"] == public["owner_profile_id"], "signed task policy or authority differs")
    need(type(manifest["sources"]) is dict and len(manifest["sources"]) == 300, "complete signed source population required")
    for expected in manifest["sources"].values():
        closed(expected, {"sha256", "executable"}, "signed source identity")
        digest(expected["sha256"])
        need(type(expected["executable"]) is bool, "exact signed source mode required")
    need(manifest["sources"]["calc.py"]["sha256"] == sha(BEFORE), "signed plus-two patch preimage differs")
    need(type(manifest["baseline_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest["baseline_commit"])
         and manifest["repository"] == "/results/native/repository"
         and manifest["profile_dir"] == "/results/native/private/profile"
         and manifest["lifecycle_dir"] == "/results/native/private/lifecycle", "signed native local owner locators differ")
    source.check_cid(manifest["repository_cid"])
    need(type(manifest["profile_content_id"]) is str
         and re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["profile_content_id"]),
         "native local profile content identity differs")
    members = public["codebase_inventory_context"]["scan"]["members"]
    for member in members:
        if member["source_cid"] is not None:
            source.check_cid(member["source_cid"], source=True)
            encoded = member["source_cid"][1:]
            raw_digest = base64.b32decode(encoded.upper() + "=" * (-len(encoded) % 8))[-32:].hex()
            need(manifest["sources"][member["path"]]["sha256"] == raw_digest,
                 "signed baseline source SHA differs from complete captured membership")
    requirement = next(member for member in members if member["path"] == "README.md")
    need(public["source_path"] == "README.md" and public["source_sha256"] == manifest["sources"]["README.md"]["sha256"]
         and type(public["source_bytes"]) is int and public["source_bytes"] == requirement["source_size_bytes"]
         and 0 < public["source_bytes"] <= 32_768, "signed complete public requirements source differs")
    roots = manifest["planning_roots"]
    closed(roots, {"request_cid", "program_root", "scan_cid"}, "native planning roots")
    for root in roots.values():
        source.check_cid(root)
    expected_graph, expected_specs = authored_graph(roots)
    need(same(graph, expected_graph) and same(manifest["tasks"], expected_specs), "independently authored full native graph/spec preimages differ")
    goal = graph["goals"][0]
    scan = public["codebase_inventory_context"]["scan"]
    need(roots["request_cid"] == structured({"schema": "authored-successor-format-work-request@1", "objective": goal["objective"],
        "task_keys": ["SUCCESSOR-TYPE", "SUCCESSOR-FORMAT"]})
        and roots["scan_cid"] == structured({"head": scan["head"], "completed_scan": scan["completion_cid"]}),
        "authored request or complete scan root differs")
    tree = structured({"schema": "supervisor-local-source-tree@1", "sources": manifest["sources"]})
    need(planning["source_tree_id"] == tree and same(planning["pending_requirements"], authored_pending(graph))
         and len(planning["pending_requirements"]) == 4, "all native task tests and goal reviews must remain exactly pending")
    need(same(planning["declared_input_closure"], declared_closure(planning["graph_cid"], tree, manifest)),
         "full signed descriptive input closure preimage differs")
    for key in ("plan_id", "manifest_cid", "graph_cid", "pending_cid"):
        source.check_cid(planning[key])
    evidence = planning["plan_evidence"]
    closed(evidence, {"schema", "validator_version", "status", "outcome", "plan_id", "plan_check_only", "bounds",
        "checks_performed", "consistency_level", "countermodel", "evidence", "findings", "formula_ids", "assumptions"},
        "retained bounded plan validation")
    need(evidence["schema"] == "ipfs_accelerate_py/agent-supervisor/formal-plan-validation@1"
         and type(evidence["validator_version"]) is int and evidence["validator_version"] == 1
         and evidence["status"] == evidence["outcome"] == "consistent"
         and evidence["plan_id"] == planning["plan_id"] and evidence["plan_check_only"] is True
         and evidence["consistency_level"] == "bounded_consistent" and evidence["countermodel"] is None
         and evidence["evidence"] == evidence["findings"] == [], "retained planning evidence scope differs")
    need(type(evidence["formula_ids"]) is list and len(evidence["formula_ids"]) == 5
         and evidence["formula_ids"] == sorted(set(evidence["formula_ids"])), "bounded formula identity set differs")
    for formula in evidence["formula_ids"]:
        source.check_cid(formula)
    bounds = evidence["bounds"]
    closed(bounds, {"schema", "configured", "domain_sizes", "effective_trace_bound", "plan_trace_bound",
        "search_nodes_explored", "truncated_dimensions"}, "retained plan validation bounds")
    need(bounds["schema"] == "ipfs_accelerate_py/agent-supervisor/formal-plan-validation-bounds@1"
         and bounds["truncated_dimensions"] == [] and bounds["effective_trace_bound"] == bounds["plan_trace_bound"] == 16
         and same(bounds["domain_sizes"], {"actors": 2, "events": 6, "evidence_requirements": 4, "fluents": 3,
            "formulas": 5, "goals": 1, "norms": 2, "provider_evidence": 0, "subgoals": 0, "tasks": 2,
         "temporal_constraints": 1}), "retained plan complete task/pending domain differs")


def verify_staged_seed_binding(reader, boundary, successor):
    """Bind the third private denial to a retained historical staging locator.

    Only this ordinary receipt is read. Its declared staged directory and the
    original archive remain unopened. These joins establish no source origin,
    current deployment, native owner, or independent transport qualification.
    """
    raw = reader.raw("source-dispatch-seed.json", 16 * MIB)
    seed = parse(raw)
    closed(seed, STAGED_FIELDS, "retained source dispatch staged receipt")
    need(seed["schema"] == "source-successor-dispatch-staged-setup@1" and seed["qualified"] is True
         and seed["fresh_native_receiving_required"] is True
         and all(seed[field] is False for field in ("native_owners_opened", "proof_authority",
             "source_execution_attested", "scan_execution_attested")), "historical staged receipt authority differs")
    locator = seed["staged_destination"]
    need(type(locator) is str and 0 < len(locator.encode("utf-8")) <= 8192
         and "\0" not in locator and not any(ord(character) < 32 for character in locator), "bounded staged seed locator required")
    path = PurePosixPath(locator)
    need(path.is_absolute() and path.name == "setup-seed" and ".." not in path.parts
         and path.as_posix() == locator, "canonical absolute setup-seed locator required")
    need(boundary["owner_private_paths"] == ["/opt/ipfs-supervisor/state", "/results/native/private", locator]
         and len(set(boundary["owner_private_paths"])) == 3, "all three root-controlled private paths must join staged locator")
    for field, expected in (("inherited_setup_epochs", 2), ("inherited_scan_pages", 10), ("inherited_reference_pages", 1),
            ("new_fitting_epochs", 0), ("new_scan_pages", 0)):
        need(type(seed[field]) is int and seed[field] == expected, "historical staged numerical scope differs: " + field)
    for field, expected in (("previous_head", successor["previous_head"]), ("current_head", successor["current_head"]),
            ("root_cid", successor["root_cid"]), ("completion_cid", successor["completion_cid"]),
            ("selection_cid", successor["selection_cid"]), ("source_delta_cid", successor["source_delta_cid"]),
            ("previous_version_id", successor["previous_model"]["version_id"]), ("selected_version_id", successor["model"]["version_id"])):
        need(same(seed[field], expected), "historical staged signed successor binding differs: " + field)
    return {"path": "source-dispatch-seed.json", "sha256": sha(raw), "bytes": len(raw), "staged_destination": locator,
        "declared_staged_directory_opened": False, "original_source_archive_opened": False,
        "transport_qualification_reperformed_here": False}


def _verify_authored_worker_receipt(output: Path, *, prepared: dict, task_cid: str) -> dict:
    """Join one real authored-worker stdout line after native STOP.

    This pure receiving gate reads only bounded ordinary files and public-key
    signatures. It grants no completion, freshness, model-owner or private
    evidence-epoch authority. The caller must hold its native lifecycle lease
    until actual STOP/UID cleanup; this function does not perform that cleanup.
    """
    reader = Reader(output, 120, native_output=True)
    closed(prepared, {"artifact", "sha256", "task_cid", "context_cid", "manifest_cid", "source_path", "source_sha256",
        "source_bytes", "completion_authority", "scope_expansion_authority", "inventory_context_cid",
        "codebase_inventory_context_cid", "codebase_successor_context_cid", "successor_selection_cid", "source_delta_cid"},
        "prepared successor task descriptor")
    need(prepared["task_cid"] == task_cid and prepared["completion_authority"] is prepared["scope_expansion_authority"] is False,
         "exact prepared task descriptor required")
    original = reader.raw(prepared["artifact"], 16 * 1024**2)
    public = parse(original)
    need(sha(original) == prepared["sha256"] and not reader.path(prepared["artifact"]).stat().st_mode & 0o222,
         "original prepared artifact bytes/mode differ")
    need(public.get("schema") == "supervisor-public-instruction@4" and public["task_cid"] == task_cid
         and public["context_cid"] == prepared["context_cid"]
         and public["context_cid"] == structured({key: value for key, value in public.items() if key != "context_cid"}),
         "public inventory artifact identity differs")
    manifest = signature(public["manifest"])
    plan = public["inventory_plan_admission"]
    planning = signature(plan["receipt"])
    need(manifest["schema"] == "supervisor-local-benchmark-manifest@6"
         and planning["schema"] == "supervisor-local-planning-receipt@4"
         and public["manifest"]["binding"]["identity"] == plan["receipt"]["binding"]["identity"] == public["owner_identity"]
         and public["manifest"]["binding"]["profile_id"] == plan["receipt"]["binding"]["profile_id"] == public["owner_profile_id"],
         "exact historical inventory public signature profiles differ")
    successor_context = public["codebase_successor_context"]
    successor = verify_successor_context(successor_context)
    context, graph = public["codebase_inventory_context"], plan["graph"]
    need(same(context, successor_context["inventory"]), "public successor inventory contexts differ")
    scan = context["scan"]
    false_authority(context["authority"]); false_authority(scan["authority"])
    need(context["schema"] == "supervisor-codebase-inventory-context@1" and context["evidence"] is None
         and scan["root_cid"] == structured(scan["root_record"])
         and scan["completion_cid"] == structured(scan["completion_record"])
         and scan["membership_cid"] == structured(scan["members"])
         and len(scan["members"]) == 300 and scan["coverage"]["inventory_entries"] == 300,
         "full authored inventory context identity/300-member closure differs")
    full_cid = structured(context)
    successor_cid = structured(successor_context)
    need(same(manifest["codebase_successor_context"], successor)
         and successor_cid == prepared["codebase_successor_context_cid"]
         and successor["selection_cid"] == prepared["successor_selection_cid"]
         and successor["source_delta_cid"] == prepared["source_delta_cid"],
         "signed/prepared successor context declaration differs")
    tasks = {task["task_key"]: task for task in graph["tasks"]}
    task_cids = {key: prompt_task_record_cid(task) for key, task in tasks.items()}
    need(set(tasks) == {"SUCCESSOR-TYPE", "SUCCESSOR-FORMAT"}
         and task_cids["SUCCESSOR-FORMAT"] == task_cid
         and tasks["SUCCESSOR-FORMAT"]["dependency_task_cids"] == [task_cids["SUCCESSOR-TYPE"]],
         "authored public task population/dependency differs")
    need(planning["manifest_cid"] == structured(public["manifest"]) == public["manifest_cid"] == prepared["manifest_cid"]
         and planning["graph_cid"] == structured(semantic(graph))
         and planning["administrator_task_cids"] == sorted(task_cids.values())
         and planning["pending_cid"] == structured(planning["pending_requirements"])
         and planning["codebase_inventory_context_cid"] == full_cid == prepared["codebase_inventory_context_cid"]
         and planning["codebase_successor_context_cid"] == successor_cid
         and planning["successor_selection_cid"] == successor["selection_cid"]
         and planning["source_delta_cid"] == successor["source_delta_cid"]
         and planning["current_facts"] == planning["removed_task_cids"] == []
         and planning["runtime_requirements_preserved"] is True,
         "signed public planning/full-context bindings differ")
    compact = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    compact["member_paths"] = [row["path"] for row in scan["members"]]
    need(same(manifest["codebase_inventory_context"], {"schema": "supervisor-codebase-inventory-declaration@1",
        "full_context_cid": full_cid, "scan": compact, "evidence": None, "authority": context["authority"]})
        and set(manifest["sources"]) == set(compact["member_paths"]), "full signed source/context declaration differs")
    spec = next(item for item in manifest["tasks"] if item["task_key"] == "SUCCESSOR-FORMAT")
    advisory = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    advisory["selected_task_members"] = [row for row in scan["members"] if row["path"] in spec["scope_paths"]]
    projection = {"schema": "supervisor-codebase-successor-worker-context@1", "task_cid": task_cid,
        "task_id": "SUCCESSOR-FORMAT", "manifest_cid": public["manifest_cid"], "graph_cid": planning["graph_cid"],
        "planning_receipt_cid": structured(plan["receipt"]), "codebase_inventory_context_cid": full_cid,
        "codebase_successor_context_cid": successor_cid, "codebase_successor": successor,
        "scan": advisory, "evidence": None, "administrator_task_cids": sorted(task_cids.values()), "task_spec": spec,
        "dependency_task_cids": tasks["SUCCESSOR-FORMAT"]["dependency_task_cids"],
        "pending_requirements": planning["pending_requirements"], "pending_cid": planning["pending_cid"],
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
        "publication_authority": False, "scope_expansion_authority": False, "authority": context["authority"]}
    need(structured(projection) == prepared["inventory_context_cid"], "actual worker projection identity differs")
    verify_authored_plan(public, manifest, graph, planning, task_cids)
    launch = reader.path("private/launch")
    need(launch.is_dir(), "native private launch logs unavailable")
    matches, boundaries, searched, entries = [], [], [], 0
    for directory, names, files in os.walk(launch, followlinks=False):
        reader.tick()
        names.sort(); files.sort()
        need(not any((Path(directory) / name).is_symlink() for name in names), "launch log directory alias")
        entries += len(names) + len(files)
        need(entries <= 4096, "bounded native log traversal required")
        if "implementation-logs" not in Path(directory).parts:
            continue
        for name in files:
            if not name.endswith(".log"):
                continue
            need(len(searched) < 64, "native implementation log count exceeds bound")
            relative = (Path(directory) / name).relative_to(reader.root).as_posix()
            raw = reader.raw(relative, 16 * 1024**2)
            pin = {"path": relative, "bytes": len(raw), "sha256": sha(raw)}
            searched.append(pin)
            for line_number, line in enumerate(raw.splitlines(), 1):
                if (b"source-successor-authored-native-worker@1" not in line
                        and b"successor-authored-worker-boundary@1" not in line):
                    continue
                try:
                    receipt = parse(line)
                except (ValueError, UnicodeError):
                    continue
                if type(receipt) is dict and receipt.get("schema") == "source-successor-authored-native-worker@1":
                    matches.append({"log": pin, "line": line_number, "receipt": receipt})
                elif type(receipt) is dict and receipt.get("schema") == "successor-authored-worker-boundary@1":
                    boundaries.append({"log": pin, "line": line_number, "receipt": receipt})
    need(len(matches) == 1, "exactly one actual authored worker stdout receipt required; logs="
         + ",".join(row["path"] for row in searched))
    found, receipt = matches[0], matches[0]["receipt"]
    need(len(boundaries) == 1 and boundaries[0]["log"] == found["log"] and boundaries[0]["line"] < found["line"],
         "one actual deployer-boundary stdout receipt must precede authored stdout in the same log")
    boundary_found = boundaries[0]
    boundary_receipt = boundary_found["receipt"]
    closed(boundary_receipt, {"schema", "boundary", "boundary_artifact", "boundary_sha256", "uid", "euid", "pid",
        "gid", "groups", "provider_calls", "training_steps", "workspace", "private_access"}, "actual successor deployer boundary stdout")
    boundary_raw = reader.raw("container-boundary.json", 16_384)
    boundary = parse(boundary_raw)
    closed(boundary, {"schema", "owner_uid", "worker_uid", "single_worker", "container_id", "image_id", "namespaces",
        "allowed_worktree_roots", "validation_repository_roots", "owner_private_paths"}, "retained root boundary")
    need(boundary["schema"] == "supervisor-container-worker-boundary@1"
         and boundary["owner_uid"] == 1000 and boundary["worker_uid"] == 1001 and boundary["single_worker"] is True
         and re.fullmatch(r"[0-9a-f]{64}", boundary["container_id"]) is not None
         and re.fullmatch(r"sha256:[0-9a-f]{64}", boundary["image_id"]) is not None
         and boundary["allowed_worktree_roots"] == ["/opt/ipfs-supervisor/worktrees"],
         "copied root-deployer container boundary differs")
    staged_seed = verify_staged_seed_binding(reader, boundary, successor)
    need(boundary["validation_repository_roots"] == [public["repository"]]
         and type(boundary["namespaces"]) is dict and set(boundary["namespaces"]) == {"mnt", "net", "pid"}
         and all(type(value) is str and re.fullmatch(name + r":\[[1-9][0-9]*\]", value)
                 for name, value in boundary["namespaces"].items()), "original retained boundary roots/namespaces differ")
    expected_boundary = {key: boundary[key] for key in ("schema", "container_id", "image_id", "namespaces", "owner_uid", "worker_uid")}
    expected_boundary.update(purpose="coding", manifest_sha256=sha(boundary_raw), completion_authority=False)
    need(same(boundary_receipt["boundary"], expected_boundary)
         and boundary_receipt["boundary_artifact"] == "/opt/ipfs-supervisor/container-boundary.json"
         and boundary_receipt["boundary_sha256"] == sha(boundary_raw)
         and type(boundary_receipt["uid"]) is int and type(boundary_receipt["euid"]) is int
         and boundary_receipt["uid"] == boundary_receipt["euid"] == 1001
         and type(boundary_receipt["pid"]) is int and boundary_receipt["pid"] == receipt["pid"]
         and type(boundary_receipt["gid"]) is int and boundary_receipt["gid"] > 0
         and type(boundary_receipt["groups"]) is list and all(type(item) is int and item > 0 for item in boundary_receipt["groups"])
         and type(boundary_receipt["provider_calls"]) is int and boundary_receipt["provider_calls"] == 0
         and type(boundary_receipt["training_steps"]) is int and boundary_receipt["training_steps"] == 0,
         "actual root-deployer boundary receipt does not join authored worker identity")
    workspace = PurePosixPath(boundary_receipt["workspace"])
    need(workspace.is_absolute() and ".." not in workspace.parts and workspace != PurePosixPath("/opt/ipfs-supervisor/worktrees")
         and workspace.is_relative_to("/opt/ipfs-supervisor/worktrees"), "actual worker workspace is outside allocated root")
    access = boundary_receipt["private_access"]
    need(type(access) is dict and set(access) == set(boundary["owner_private_paths"]) and len(access) == 3
         and all(type(row) is dict and set(row) == {"read", "write", "execute"}
                 and all(value is False for value in row.values()) for row in access.values()),
         "actual worker private-authority denial observations are incomplete")
    closed(receipt, {"schema", "status", "pid", "uid", "task_cid", "path", "before_sha256", "after_sha256",
        "public_instruction", "provider_calls", "training_steps", "proof_authority", "completion_authority",
        "native_completion_recorded_here"}, "authored successor worker stdout")
    need(type(receipt["uid"]) is int and receipt["uid"] == 1001 and type(receipt["pid"]) is int and receipt["pid"] > 1
         and receipt["task_cid"] == task_cid and receipt["status"] == "materialized" and receipt["path"] == "calc.py"
         and receipt["before_sha256"] == sha(BEFORE) and receipt["after_sha256"] == sha(AFTER)
         and type(receipt["provider_calls"]) is int and receipt["provider_calls"] == 0
         and type(receipt["training_steps"]) is int and receipt["training_steps"] == 0
         and receipt["proof_authority"] is False and receipt["completion_authority"] is False
         and receipt["native_completion_recorded_here"] is False, "real authored worker UID/task/patch/accounting differs")
    inclusion = receipt["public_instruction"]
    closed(inclusion, {"schema", "artifact", "artifact_sha256", "context_cid", "task_cid", "task_id", "manifest_cid",
        "source_path", "source_sha256", "source_bytes", "block_sha256", "block_bytes", "manifest_signature_verified",
        "verbatim_utf8", "semantic_minification_applied", "source_freshness_verified", "historical_replay",
        "completion_authority", "scope_expansion_authority", "extra_provider_calls", "codebase_inventory", "codebase_successor"},
        "actual public successor inclusion")
    need(inclusion["schema"] == "supervisor-public-instruction-inclusion@4"
         and inclusion["artifact"] == prepared["artifact"] and inclusion["artifact_sha256"] == prepared["sha256"]
         and inclusion["context_cid"] == prepared["context_cid"] and inclusion["task_cid"] == task_cid
         and inclusion["manifest_cid"] == prepared["manifest_cid"] and inclusion["source_path"] == prepared["source_path"] == "README.md"
         and inclusion["source_sha256"] == prepared["source_sha256"] and inclusion["source_bytes"] == prepared["source_bytes"]
         and inclusion["manifest_signature_verified"] is True and inclusion["source_freshness_verified"] is True
         and inclusion["historical_replay"] is False and inclusion["verbatim_utf8"] is True
         and inclusion["semantic_minification_applied"] is False and type(inclusion["extra_provider_calls"]) is int
         and inclusion["extra_provider_calls"] == 0 and inclusion["completion_authority"] is False
         and inclusion["scope_expansion_authority"] is False, "real public reader inclusion differs")
    need(inclusion["task_id"] == "SUCCESSOR-FORMAT" and type(inclusion["block_bytes"]) is int
         and 0 < inclusion["block_bytes"] <= 16 * MIB and type(inclusion["source_bytes"]) is int
         and 0 < inclusion["source_bytes"] <= 32_768, "bounded public inclusion byte accounting differs")
    digest(inclusion["block_sha256"])
    inventory = inclusion["codebase_inventory"]
    need(same(inventory, {"context_cid": prepared["inventory_context_cid"], "codebase_inventory_context_cid": full_cid,
        "root_cid": scan["root_cid"], "completion_cid": scan["completion_cid"], "membership_cid": scan["membership_cid"],
        "planning_receipt_cid": structured(plan["receipt"]), "administrator_task_cids": sorted(task_cids.values()),
        "pending_cid": planning["pending_cid"], "current_facts": [], "removed_task_cids": [],
        "runtime_requirements_preserved": True, "native_inventory_current_verified_here": False,
        "native_persistence_verified_here": False, "authority": context["authority"]}),
        "real worker public advisory/full-population inclusion differs")
    need(same(inclusion["codebase_successor"], {"context_cid": successor_cid,
        **{key: successor[key] for key in ("selection_cid", "source_delta_cid", "previous_head", "current_head",
            "root_cid", "completion_cid")}, "native_inventory_current_verified_here": False,
        "native_successor_current_verified_here": False, "authority": successor_context["authority"]}),
        "real worker public successor inclusion differs")
    return {"schema": "source-successor-authored-worker-receipt-verification@1", "verified": True,
        "actual_stdout_observation": found, "original_artifact": {"path": prepared["artifact"],
            "sha256": sha(original), "bytes": len(original)}, "public_signatures_verified": True,
        "actual_boundary_stdout_observation": boundary_found,
        "retained_root_boundary": {"path": "container-boundary.json", "sha256": sha(boundary_raw), "bytes": len(boundary_raw)},
        "retained_staged_seed_binding": staged_seed,
        "context_cid": full_cid, "worker_context_cid": prepared["inventory_context_cid"],
        "successor_context_cid": successor_cid, "selection_cid": successor["selection_cid"],
        "source_delta_cid": successor["source_delta_cid"],
        "task_cid": task_cid, "native_registry_opened": False, "native_inventory_freshness_verified_here": False,
        "private_evidence_epoch_verified_here": False, "process_origin_attested": False,
        "native_formal_compilation_reperformed_here": False, "numerical_execution_reperformed_here": False,
        "completion_authority": False, "proof_authority": False, "guarded_read_bytes": reader.total_read_bytes}


def verify_authored_worker_receipt(output: Path, *, prepared: dict, task_cid: str) -> dict:
    """Verify bounded retained signed-successor stdout after the caller's STOP."""
    try:
        return _verify_authored_worker_receipt(output, prepared=prepared, task_cid=task_cid)
    except (source.SourceSuccessorAuditError, KeyError, TypeError, OSError, RecursionError, UnicodeError) as error:
        raise SignedSuccessorAuditError("invalid signed successor receipt: " + str(error)) from error
