"""Read-only audit of a copied finite advisory/native-worker qualification.

This verifies retained bytes, signatures and joins. It does not reopen native
owners, infer, fit, run checkers, launch processes or grant execution authority.
Container paths are resolved only into the supplied copied output namespace.
Historical public-key signatures are not a current profile/revocation check.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys

SCHEMA = "finite-repository-advisory-join-independent-audit@1"
RESULT_SCHEMA = "finite-repository-advisory-native-worker-qualification@1"
MAX_FILE_BYTES = 64 * 1024 * 1024
INVENTORY = {"calc.py", "check_offset.py", "check_type.py", "consumer.py", "decoy.py",
             "support.py", "unsupported.py", "known_variant.py", "tune.py", "canary.py"}
CONTEXT_FALSE = {"source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "proof_authority", "execution_authority", "completion_authority", "mutation_authority",
    "admission_authority", "qualified", "formalized", "promotion_performed", "behavioral_satisfaction",
    "formal_decoder_available", "convergence_proved"}
NATIVE_FEATURE_FALSE = {"qualified", "admitted", "formalized", "promotion_performed", "proof_authority",
    "source_runtime_semantics_verified", "behavioral_satisfaction", "admission_authority", "completion_authority"}
PROPOSAL_FALSE = {"source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority", "publication_authority",
    "production_activation", "native_worker_loop_qualified", "convergence_proved",
    "filesystem_isolation", "network_isolation", "owner_keys_inaccessible",
    "process_origin_attested", "untrusted_worker_isolated"}
COLD_FIELDS = {"source_cid", "query", "domain_inputs", "observations",
               "eligible_requirement_ids", "residual_requirement_ids", "finite_selected_task_ids",
               "operation_catalog_cid", "model_status"}
ADVISORY_FIELDS = ("source_cid", "query", "domain_inputs", "domain_cid", "observations", "eligible_clause_ids",
    "residual_clause_ids", "clause_results", "selected_task_ids", "declared_task_requirement_ids",
    "candidate_task_meaning", "current_facts_count", "operation_catalog_cid")


class AdvisoryWorkerAuditError(ValueError):
    """A retained artifact or claimed qualification join does not verify."""


def _need(value, message):
    if not value:
        raise AdvisoryWorkerAuditError(message)


def _json_bytes(value, *, ascii=True):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii,
                      allow_nan=False).encode("utf-8")


def _strict(value, depth=0):
    _need(depth <= 96, "structured identity exceeds nesting limit")
    if type(value) is dict:
        _need(all(type(key) is str for key in value), "structured identity needs string keys")
        for child in value.values():
            _strict(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _strict(child, depth + 1)
    else:
        _need(type(value) in {str, int, bool, type(None)}, "structured identity rejects floats/host objects")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _cid(raw, codec="raw"):
    _need(codec in {"raw", "dag-json"}, "unreviewed CID codec")
    # CIDv1, raw=0x55 or dag-json=0x0129, sha2-256 multihash.
    prefix = b"\x01\x55" if codec == "raw" else b"\x01\xa9\x02"
    return "b" + base64.b32encode(prefix + b"\x12\x20" + hashlib.sha256(raw).digest()).decode().lower().rstrip("=")


def _structured_cid(value):
    _strict(value)
    return _cid(_json_bytes(value, ascii=False), "dag-json")


def _digest(value):
    # Numerical checkpoints use canonical finite native JSON, including floats.
    return _sha(_json_bytes(value))


def _false(value, fields, label):
    _need(type(value) is dict and fields <= set(value), label + " lacks exact authority fields")
    _need(all(value[key] is False for key in fields), label + " has an authority/isolation claim")


def _signature(envelope):
    _need(type(envelope) is dict and set(envelope) == {"payload", "binding"}, "signed envelope is not closed")
    binding = envelope["binding"]
    _need(type(binding) is dict and set(binding) == {"identity", "profile_id", "signature"},
          "signature binding is not closed")
    did = binding["identity"]
    _need(type(did) is str and did.startswith("did:key:z"), "signature needs Ed25519 did:key")
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    number = 0
    for character in did[9:]:
        _need(character in alphabet, "signature did:key is not base58btc")
        number = number * 58 + alphabet.index(character)
    encoded = did[9:]
    raw = b"\0" * (len(encoded) - len(encoded.lstrip("1"))) + number.to_bytes((number.bit_length() + 7) // 8, "big")
    _need(len(raw) == 34 and raw[:2] == b"\xed\x01", "signature did:key is not Ed25519")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(raw[2:]).verify(
            base64.b64decode(binding["signature"], validate=True), _json_bytes(envelope["payload"]))
    except Exception as error:
        raise AdvisoryWorkerAuditError("retained public-key signature does not verify") from error
    return envelope["payload"]


class _Reader:
    def __init__(self, namespace):
        self.root = Path(namespace).absolute()
        _need(self.root.resolve(strict=True) == self.root and self.root.is_dir(), "canonical copied namespace required")
        self.observed = {}

    def path(self, value):
        _need(type(value) is str and value and "\0" not in value, "invalid artifact locator")
        source = PurePosixPath(value)
        _need(".." not in source.parts, "artifact locator escapes namespace")
        mappings = {"/results": self.root, "/opt/ipfs-supervisor/source": self.root / "source",
                    "/opt/ipfs-supervisor/datasets": self.root / "datasets",
                    "/opt/ipfs-supervisor/kit": self.root / "kit",
                    "/opt/ipfs-supervisor/finite-handoffs": self.root / "handoffs"}
        if source.is_absolute():
            if Path(value).is_relative_to(self.root):
                path = Path(value)
            else:
                path = None
                for prefix, destination in mappings.items():
                    if source == PurePosixPath(prefix) or source.is_relative_to(prefix):
                        path = destination / source.relative_to(prefix)
                        break
                _need(path is not None, "artifact locator is outside copied container mounts: " + value)
        else:
            path = self.root / source
        _need(path.is_relative_to(self.root), "mapped artifact escapes copied namespace")
        for parent in (path, *path.parents):
            _need(not parent.is_symlink(), "copied artifact contains symlink: " + str(path))
            if parent == self.root:
                break
        return path

    def raw(self, value):
        path = self.path(str(value))
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            _need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES,
                  "artifact is not a bounded regular file: " + str(path))
            chunks, total = [], 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                _need(total <= MAX_FILE_BYTES, "artifact grew beyond byte bound")
            after = os.fstat(descriptor)
            current = path.stat(follow_symlinks=False)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
            _need(identity(before) == identity(after) == identity(current), "artifact changed during read")
            raw = b"".join(chunks)
            observation = {"bytes": len(raw), "sha256": _sha(raw)}
            _need(path not in self.observed or self.observed[path] == observation, "artifact changed between checks")
            self.observed[path] = observation
            return raw
        finally:
            os.close(descriptor)

    def json(self, value):
        raw = self.raw(value)
        def closed(pairs):
            result = {}
            for key, child in pairs:
                _need(key not in result, "JSON artifact has duplicate keys")
                result[key] = child
            return result
        return json.loads(raw, object_pairs_hook=closed,
                          parse_constant=lambda token: (_ for _ in ()).throw(AdvisoryWorkerAuditError("nonfinite JSON: " + token)))

    def pin(self, value):
        _need(type(value) is dict and {"path", "sha256"} <= set(value), "complete raw artifact pin required")
        raw = self.raw(value["path"])
        size = value.get("bytes", value.get("size_bytes"))
        _need(type(size) is int and size == len(raw) and value["sha256"] == _sha(raw), "raw artifact pin differs")
        if "cid" in value:
            _need(value["cid"] == _cid(raw), "raw artifact CID differs")
        return raw

    def unchanged(self):
        for path, pin in tuple(self.observed.items()):
            _need(_sha(self.raw(str(path))) == pin["sha256"], "audit changed an observed artifact")


def _context(reader, path):
    directory = reader.path(path)
    binding_raw = reader.raw(str(directory / "context.json"))
    binding = reader.json(str(directory / "context.json"))
    _need(binding_raw == _json_bytes(binding), "context JSON is not canonical native JSON")
    _need(binding["schema"] == "supervisor-codebase-feature-context@1"
          and binding["profile"] == "codebase_ir/source_bound_feature_v1", "unexpected advisory context profile")
    _need(binding["context_cid"] == _structured_cid({key: child for key, child in binding.items() if key != "context_cid"}),
          "advisory context CID differs")
    _false(binding["authority"], CONTEXT_FALSE, "advisory context")
    _need(set(binding["authority"]) == CONTEXT_FALSE, "advisory authority inventory differs")
    mode = binding["mode"]
    _need(mode in {"model_off", "frozen", "train"} and binding["model_enabled"] is (mode != "model_off"),
          "advisory context mode differs")
    roles = {"invocation"} if mode == "model_off" else {"invocation", "record", "checkpoint", "lineage", "inference", "metrics"}
    _need(set(binding["artifacts"]) == roles and {item.name for item in directory.iterdir()}
          == {"context.json", *(role + ".json" for role in roles)}, "advisory retained artifact inventory differs")
    retained = {}
    for role, row in binding["artifacts"].items():
        _need(row["role"] == role and row["relative_path"] == role + ".json", "advisory role locator differs")
        raw = reader.raw(str(directory / row["relative_path"]))
        _need(row["sha256"] == _sha(raw) and row["size_bytes"] == len(raw), "advisory retained bytes differ")
        _need(row["blob_cid"] == _structured_cid({key: child for key, child in row.items() if key != "blob_cid"}),
              "advisory retained blob identity differs")
        retained[role] = reader.json(str(directory / row["relative_path"]))
        _need(_json_bytes(retained[role]) == raw, "advisory numerical artifact is not canonical")
    invocation = retained["invocation"]
    _need(set(invocation) == {"schema", "mode", "head", "version_id", "operation_id", "parent_version_id", "selections", "configuration"}
          and invocation["mode"] == mode and invocation["head"] == binding["head"], "closed native feature invocation differs")
    if mode == "model_off":
        _need(binding["version_id"] is None and binding["actual_training_delta"] == 0, "off context carries fitted state")
        _need(invocation["version_id"] is invocation["parent_version_id"] is invocation["configuration"] is None
              and invocation["selections"] == [] and invocation["operation_id"] == "", "off context carries fitting controls")
        return binding, retained
    record, checkpoint, lineage = retained["record"], retained["checkpoint"], retained["lineage"]
    state, report = checkpoint["state"], checkpoint["report"]
    _false(record["authority"], NATIVE_FEATURE_FALSE, "native feature record")
    _need(set(record["authority"]) == NATIVE_FEATURE_FALSE, "native feature authority inventory differs")
    _false(retained["inference"]["authority"], NATIVE_FEATURE_FALSE, "native current inference")
    _need(binding["model_record_cid"] == _structured_cid(record)
          and binding["candidate_checkpoint_raw_cid"] == record["checkpoint_raw_cid"] == _cid(_json_bytes(checkpoint)),
          "native record/checkpoint raw identity differs")
    _need(record["head"] == binding["head"] == report["codebase_provenance"]["head"], "model source head differs")
    _need(record["version_id"] == binding["version_id"] and state["latent_width"] == binding["latent_width"] == 8,
          "advisory selected version/width differs")
    _need(binding["parameter_dtype"] == "float64" and binding["state_sha256"] == _digest(state)
          and binding["feature_space_sha256"] == _digest(checkpoint["feature_space"]), "advisory numerical identity differs")
    _need(retained["inference"]["head"] == binding["head"] and retained["inference"]["version_id"] == binding["version_id"]
          and retained["inference"]["training_executed"] is False, "advisory inference source/version/training differs")
    _need(type(lineage) is list and 1 <= len(lineage) <= 8 and lineage[0]["checkpoint"] == checkpoint,
          "complete advisory lineage is missing")
    _need(lineage[0]["version"]["version_id"] == binding["version_id"], "advisory lineage starts elsewhere")
    for position, row in enumerate(lineage):
        version, saved = row["version"], row["checkpoint"]
        _false(saved["report"]["codebase_provenance"], NATIVE_FEATURE_FALSE, "ancestral native feature provenance")
        _false(saved["report"], {"qualified", "admitted", "formalized", "promotion_performed"}, "ancestral numerical report")
        raw = _json_bytes(saved)
        _need(version["artifact"]["sha256"] == _sha(raw) and version["artifact"]["bytes"] == len(raw),
              "ancestral checkpoint differs from registry artifact identity")
        parent = lineage[position + 1] if position + 1 < len(lineage) else None
        _need(version["parent_version_id"] == (None if parent is None else parent["version"]["version_id"]),
              "advisory lineage parent ordering differs")
        if parent is not None:
            left, right = parent["checkpoint"], saved
            a, b = left["report"]["codebase_provenance"], right["report"]["codebase_provenance"]
            _need(right["report"]["base_state_sha256"] == _digest(left["state"])
                  and right["contract"] == left["contract"] and right["feature_space"] == left["feature_space"]
                  and right["state"]["optimizer_config"] == left["state"]["optimizer_config"],
                  "advisory child does not bind exact parent basis/optimizer")
            _need(b["continuation"] == "exact_frozen_basis_adam_resume"
                  and all(a[key] == b[key] for key in ("selections", "tuning_targets", "canary_targets", "replay_targets")),
                  "advisory child changed ancestral splits or fixed evaluation targets")
    _need(binding["actual_training_delta"] == (report["attempted_epochs"] if mode == "train" else 0),
          "advisory context fitting cost differs")
    if mode == "frozen":
        _need(invocation["version_id"] == binding["version_id"] and invocation["configuration"] is None
              and invocation["parent_version_id"] is None and invocation["selections"] == [] and invocation["operation_id"] == "",
              "frozen context carries fitting controls")
    else:
        _need(invocation["version_id"] is None and invocation["parent_version_id"] == record["parent_version_id"]
              and invocation["selections"] == report["codebase_provenance"]["selections"]
              and invocation["configuration"] == {key: report["configuration"][key] for key in ("epochs", "learning_rate", "seed")},
              "training invocation differs from native selected model controls")
    _need(binding["selected_total_epochs"] == state["completed_epochs"]
          and binding["selected_optimizer_steps"] == [item["step"] for item in state["adam"]],
          "advisory selected optimizer counts differ")
    return binding, retained


def _admission(reader, path, *, offset, no_work):
    admission = reader.json(path)
    _need(set(admission) == {"declaration", "evidence", "graph", "local_admission", "receipt"},
          "finite admission is not complete")
    declared = _signature(admission["declaration"])
    receipt = _signature(admission["receipt"])
    manifest = _signature(declared["manifest"])
    graph, evidence = admission["graph"], admission["evidence"]
    _need(receipt["declaration_cid"] == _structured_cid(admission["declaration"])
          and receipt["evidence_cid"] == _structured_cid(evidence)
          and receipt["graph_cid"] == _structured_cid(graph), "signed admission joins differ")
    _need(receipt["schema"] == "supervisor-finite-repository-admission@1"
          and receipt["profile"] == "finite-repository-fixed-administrator-population@1"
          and receipt["policy"]["models"] == "off", "worker admission widened model-off policy")
    _false(receipt, {"source_semantics_verified", "runtime_behavior_verified", "proof_authority",
        "code_proof_authority", "production_admitted", "production_activation", "execution_authority",
        "completion_authority", "mutation_authority", "omission_authority", "worker_launched", "convergence_proved"},
        "finite admission receipt")
    _need(receipt["planning_permitted"] is (not no_work) and receipt["no_work_review_only"] is no_work
          and (admission["local_admission"] is None) is no_work, "no-work admission authority differs")
    _need(receipt["local_admission_cid"] == (None if no_work else _structured_cid(admission["local_admission"])),
          "signed local admission join differs")
    if not no_work:
        _signature(admission["local_admission"]["manifest"])
        _signature(admission["local_admission"]["receipt"])
        _need(admission["local_admission"]["graph"] == graph, "local admission changed original graph")
    tasks = {row["task_key"]: row for row in graph["tasks"]}
    specs = {row["task_key"]: row for row in manifest["tasks"]}
    _need(set(tasks) == set(specs) == {"FINITE-TYPE", "FINITE-OFFSET"}, "original two-task population differs")
    _need(tasks["FINITE-TYPE"]["dependency_task_cids"] == []
          and tasks["FINITE-OFFSET"]["dependency_task_cids"] == [tasks["FINITE-TYPE"]["content_id"]],
          "signed task dependency meaning differs")
    for key, script in (("FINITE-TYPE", "check_type.py"), ("FINITE-OFFSET", "check_offset.py")):
        _need(specs[key]["validations"][0]["argv"] == ["python3", script]
              and tasks[key]["validations"][0]["argv"] == ["python3", script]
              and {row["path"] for row in specs[key]["outputs"]} == {"calc.py"},
              "signed task public validation/output meaning differs")
    semantic = receipt["semantic_context"]
    _need(receipt["semantic_context_cid"] == _structured_cid(semantic), "signed semantic context CID differs")
    _need(semantic["administrator_task_cids"] == sorted(row["content_id"] for row in tasks.values())
          and semantic["task_population_preserved"] is True and semantic["finite_facts_are_context_only"] is True,
          "semantic context lost signed original tasks")
    _need(semantic["domain_inputs"] == [-2, -1, 0, 1, 2]
          and semantic["query"]["requirement_ids"] == ["finite-integer-offset-goal", "finite-integer-type-goal"],
          "complete two-clause/five-input finite meaning differs")
    expected_observations = [{"input": item, "input_type": "int", "output": item + offset, "output_type": "int"}
                             for item in [-2, -1, 0, 1, 2]]
    _need(semantic["observations"] == expected_observations
          == evidence["match"]["observation"]["observations"], "finite retained observations differ from exact source fixture")
    _need(semantic["head"] == declared["head"] == evidence["match"]["head"], "admission current head joins differ")
    _need(semantic["source_cid"] == evidence["match"]["source_cid"]
          and semantic["operation_catalog_cid"] == _structured_cid(declared["operation_catalog"]),
          "admission source/catalog identity differs")
    _need(semantic["residual_requirement_ids"] == ([] if no_work else ["finite-integer-offset-goal"])
          and semantic["finite_selected_task_ids"] == ([] if no_work else ["task:finite:offset"]),
          "admission finite residual differs")
    _need(evidence["training_steps_during_preview"] == evidence["planning_model_calls"] == 0
          and evidence["feature_context"]["mode"] == "model_off", "admission performs training/model planning")
    return admission, semantic, manifest, tasks


def _capture(reader, path, head, *, final):
    captured = reader.json(path)
    _need(captured["heads"] == [head] and len(captured["structural_manifests"]) == 1,
          "full captured source head/manifest missing")
    manifest = captured["structural_manifests"][0]
    _need(_structured_cid(manifest) == head["manifest_cid"], "full captured manifest CID differs")
    entries = {row["path"]: row for row in captured["sources"]}
    _need(set(entries) == INVENTORY and len(captured["sources"]) == len(entries)
          and manifest["snapshot"]["entries"] == captured["sources"], "full ten-unit source inventory differs")
    ast = {row["provenance"]["path"]: row for row in captured["ast"]}
    _need(set(ast) == INVENTORY and len(captured["ast"]) == len(ast), "full ten-unit AST inventory differs")
    _need({"imports", "calls"} <= {row["relation"] for row in captured["kg"]}, "full capture lacks actual import/call edges")
    units = {row["source_key"]: row for row in manifest["units"]}
    for name, entry in entries.items():
        source = reader.raw("native/cas/source/" + entry["source_cid"][:4] + "/" + entry["source_cid"])
        _need(_cid(source) == entry["source_cid"] and len(source) == entry["size_bytes"], "captured source raw CID/size differs")
        _need(ast[name]["provenance"]["source_cid"] == entry["source_cid"]
              and units["raw:" + name.encode().hex()]["ast_cid"] == _structured_cid(ast[name]),
              "captured AST source/CID correspondence differs")
        if final:
            _need(reader.raw("native/repository/" + name) == source, "published repository differs from successor capture")
    _need(entries["calc.py"]["source_cid"] != entries["decoy.py"]["source_cid"], "same-name decoy selected as actual source")
    if "compiled_logic" in captured:
        compiled = {row["source_path"]: row for row in captured["compiled_logic"]}
        _need(set(compiled) == INVENTORY, "full captured frontend dispositions missing")
        for name, row in compiled.items():
            _need(row["source_cid"] == entries[name]["source_cid"] and row["solver_executed"] is False
                  and row["source_semantics_verified"] is False and row["proof_authority"] is False,
                  "captured unproved frontend gained solver/source authority")
    return captured, entries


def _worker(reader, report, task_cid):
    lifecycle = reader.json("native/native-lifecycle.json")
    _need(lifecycle["start"]["status"] == lifecycle["stop"]["status"] == "succeeded"
          and lifecycle["start"]["operation"] == "start" and lifecycle["stop"]["operation"] == "stop",
          "actual native START/STOP receipts did not succeed")
    stop = lifecycle["stop"]["data"]
    cleanup = stop["isolated_worker_cleanup"]
    _need(stop["old_tree_fenced"] is True and stop["new_process_identity"] is None
          and cleanup["worker_uid"] == 1001 and cleanup["single_worker"] is True
          and cleanup["returncode"] == 0 and cleanup["completion_authority"] is False,
          "native STOP lacks exact old-tree/UID1001 cleanup")
    _need(lifecycle["remaining_processes"] == 0 and lifecycle["bootstrap_errors"] == [], "native worker retained processes/bootstrap errors")
    task = lifecycle["task"]
    _need(task["status"] == "completed" and task["revision"] >= 4, "native residual never completed")
    completion = task["body"]["completion_receipt"]
    transition = completion["validation"]["accepted_source_transition"]
    binding = transition["database_attempt_binding"]
    _need(completion["operation"] == "database_complete" and completion["validation"]["outcome"] == "passed"
          and transition["database_task_cid"] == binding["task_cid"] == task_cid
          and transition["task_alias"] == "FINITE-OFFSET" and transition["worker_self_approval"] is False
          and transition["task_completion_authority"] is False, "native portal completion/task binding differs")
    for key in ("attempt_id", "claim_id", "lease_id", "fencing_token", "fence_epoch"):
        _need(completion[key] == binding[key], "native completion claim/fence differs")
    _need(transition["baseline_ref"] == report["original_commit"]
          and transition["merge_commit"] == report["published_commit"]
          and transition["integration_commit_proof"]["passed"] is True
          and transition["declared_output_invariant"]["passed"] is True,
          "native accepted publication/baseline/output proof differs")
    _need(len(report["published_commit_parents"]) == 2
          and report["published_commit_parents"][0] == report["original_commit"]
          and report["published_commit_parents"][1] == transition["implementation_commit"],
          "native two-parent publication proof differs")
    _need(report["changed_paths"] == ["calc.py"], "worker changed unexpected repository paths")
    _need(lifecycle["observed_worker_allocations"] and all(row["task_id"] == "FINITE-OFFSET"
          and row["workspace_path"].startswith("/opt/ipfs-supervisor/worktrees/")
          for row in lifecycle["observed_worker_allocations"]), "actual isolated residual allocation missing")
    scope = _signature(report["execution_scope_after_stop"])
    _need(scope["finite_admission_cid"] == report["execution_scope_after_stop"]["payload"]["finite_admission_cid"],
          "execution scope signature payload differs")
    return lifecycle, scope


def _preview(reader, path, context, semantic, *, no_work):
    value = reader.json(path)
    _need(value["result_cid"] == _structured_cid({key: child for key, child in value.items() if key != "result_cid"}),
          "advisory preview result CID differs")
    _need(value["feature_context"] == context and value["training_steps_during_preview"] == 0
          and value["planning_model_calls"] == 0 and value["reservation_released_on_return"] is True,
          "advisory preview context/cost/release differs")
    _false(value, {"source_semantics_verified", "runtime_behavior_verified", "proof_authority", "execution_authority",
        "completion_authority", "mutation_authority", "production_admitted", "worker_launched", "convergence_proved"},
        "advisory preview")
    match = value["match"]
    _need(match["head"] == semantic["head"] and match["source_cid"] == semantic["source_cid"]
          and match["query"] == semantic["query"] and match["domain_inputs"] == semantic["domain_inputs"]
          and match["observation"]["observations"] == semantic["observations"]
          and match["eligible_clause_ids"] == semantic["eligible_requirement_ids"]
          and match["residual_clause_ids"] == semantic["residual_requirement_ids"],
          "advisory preview changed original finite semantic partition")
    _need(value["selected_task_ids"] == semantic["finite_selected_task_ids"]
          and value["operation_catalog_cid"] == semantic["operation_catalog_cid"]
          and len(value["obligation_graph"]["root_obligation_ids"]) == 2
          and value["critique"]["accepted"] is True and value["critique"]["truncated"] is False,
          "advisory preview changed original task/clause/catalog/critic meaning")
    plan = value["candidate_plan"]
    _need([row["task_id"] for row in plan["tasks"]] == value["selected_task_ids"], "advisory candidate task selection differs")
    if no_work:
        _need(plan["tasks"] == plan["effects"] == [] and value["execution_plan"]["admitted"] is False,
              "advisory no-work preview invented execution")
    else:
        _need(len(plan["tasks"]) == len(plan["effects"]) == 1
              and plan["tasks"][0]["requirement_id"] == "finite-integer-offset-goal"
              and plan["tasks"][0]["outputs"] == ["calc.py"]
              and plan["effects"][0]["target_id"] == "calc.py"
              and plan["effects"][0]["operation"] == "update", "advisory candidate changed reviewed residual effect")
    numerical = value["feature_validation_policy"]["numerical_verification_points"]
    _need(numerical == ([] if context["mode"] == "model_off" else ["entry", "after_reservation_release"]),
          "advisory numerical verification points differ")
    return value


def _training(reader, contexts, attempts_path, phases_path, criteria_path):
    attempts, phases, criteria = reader.json(attempts_path), reader.json(phases_path), reader.json(criteria_path)
    configuration = criteria["configuration"]
    _need(configuration == {"epochs": 16, "learning_rate": .01, "seed": 1729}, "predeclared training configuration differs")
    rows = attempts["attempts"]
    _need(type(rows) is list and len(rows) == 3 and all(row["status"] == "completed" for row in rows)
          and attempts["unknown_fitting_attempt_count"] == 0
          and attempts["known_actual_attempted_epochs"] == 48, "complete positive fitting-attempt accounting differs")
    records = {binding["version_id"]: retained for binding, retained in contexts.values()
               if binding["mode"] == "train"}
    _need(len(records) == 3 and {row["version_id"] for row in rows} == set(records), "fitting attempts do not name actual selected contexts")
    total = 0
    for row in rows:
        report = records[row["version_id"]]["checkpoint"]["report"]
        _need(row["requested_epochs"] == row["actual_attempted_epochs"] == report["attempted_epochs"] == 16,
              "attempted fitting cost differs from native selected checkpoint")
        _need(all(report["configuration"][key] == value for key, value in configuration.items()),
              "native fitting configuration differs from predeclared controls")
        total += row["actual_attempted_epochs"]
    _need(type(phases) is list and phases and all(type(row["wall_seconds"]) in {float, int}
          and math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0
          and row["status"] in {"completed", "failed"} for row in phases), "complete finite phase cost ledger missing")
    names = [row["phase"] for row in phases]
    _need(len(names) == len(set(names)) and {row["phase"] for row in rows} <= set(names),
          "fitting attempts missing from complete phase ledger")
    return total, phases, criteria


def _projection(value):
    match = value["match"]
    return {name: match[name] for name in ("source_cid", "query", "domain_inputs", "domain_cid",
            "eligible_clause_ids", "residual_clause_ids", "clause_results")} | {
        "observations": match["observation"]["observations"],
        "selected_task_ids": value["selected_task_ids"],
        "declared_task_requirement_ids": value["declared_task_requirement_ids"],
        "candidate_task_meaning": value["candidate_plan"]["tasks"],
        "current_facts_count": value["current_facts_count"],
        "operation_catalog_cid": value["operation_catalog_cid"]}


def _comparison(reader, path, previews):
    value = reader.json(path)
    expected = _projection(next(iter(previews.values())))
    _need(value["agreement"] is True and value["compared_fields"] == list(ADVISORY_FIELDS)
          and value["projection"] == expected and all(_projection(row) == expected for row in previews.values()),
          "retained advisory comparison does not independently replay")
    references = [{"label": label, "result_cid": row["result_cid"],
                   "feature_context_cid": row["feature_context"]["context_cid"]}
                  for label, row in previews.items()]
    _need(value["preview_records"] == references and len({row["feature_context_cid"] for row in references}) == len(references),
          "advisory comparison aliases distinct selected modes")
    _false(value, {"features_choose_fixed_edit", "execution_authority", "completion_authority"}, "advisory comparison")
    return value


def _proposal(reader, admission, semantic, contexts):
    generated = reader.json("native/generated-candidate.json")
    reviewed = reader.json("native/reviewed-candidate.json")
    review = _signature(reviewed)
    _false(generated, PROPOSAL_FALSE, "trusted generated proposal")
    _false(review, PROPOSAL_FALSE - {"untrusted_worker_isolated"}, "trusted reviewed proposal")
    _need(generated["status"] == "candidate_generated" and generated["proposal_generated"] is True
          and generated["result_cid"] == _structured_cid({key: value for key, value in generated.items() if key != "result_cid"}),
          "trusted proposal did not complete or result CID differs")
    _need(generated["parent_admission"] == admission and generated["parent_admission_cid"] == _structured_cid(admission)
          and generated["reviewed_candidate"] == reviewed and generated["reviewed_candidate_cid"] == _structured_cid(reviewed),
          "trusted proposal lost signed parent/review joins")
    _need(generated["training_steps"] == generated["provider_calls"] == 0
          and generated["canonical_source_unchanged"] is True and generated["reservation_released_on_return"] is True,
          "trusted proposal fitted, contacted provider or changed canonical source")
    child = generated["child_process"]
    _need(child["returncode"] == 0 and child["workspace_cleaned"] is True
          and all(child[key] is False for key in ("timed_out", "cancelled", "resource_exhausted", "unavailable", "output_truncated")),
          "trusted proposal child did not genuinely finish within retained bounds")
    raws = {role: reader.pin(pin) for role, pin in generated["artifacts"].items()}
    before, after = raws["source"], raws["replacement"]
    _need(before == b"def increment(n: int) -> int:\n    return n + 1\n"
          and after == b"def increment(n: int) -> int:\n    return n + 2\n", "trusted proposal replacement changed authored meaning")
    _need(review["before_sha256"] == _sha(before) and review["after_sha256"] == _sha(after)
          and review["source_cid"] == semantic["source_cid"] == _cid(before)
          and generated["replacement_cid"] == review["after_cid"] == _cid(after),
          "trusted proposal source/replacement raw CID or SHA differs")
    bridge = reader.json("native/candidate-bridge.json")
    descriptor = reader.json("native/candidate-descriptor.json")
    raw_candidate = reader.pin(bridge["public_candidate_pin"])
    candidate = reader.json(descriptor["artifact"])
    _need(raw_candidate == _json_bytes(candidate, ascii=False) and _sha(raw_candidate) == descriptor["sha256"],
          "public worker handoff is not the canonical exact retained artifact")
    _need(candidate["candidate_cid"] == descriptor["candidate_cid"]
          == _structured_cid({key: value for key, value in candidate.items() if key != "candidate_cid"}),
          "public worker candidate CID differs")
    _false(candidate, {"source_semantics_verified", "proof_authority", "execution_authority", "publication_authority",
                       "completion_authority", "task_omission_authority"}, "public worker candidate")
    _need(candidate["finite_admission"] == admission and candidate["finite_admission_cid"] == _structured_cid(admission)
          and candidate["original_clause_ids"] == semantic["query"]["requirement_ids"]
          and candidate["original_prompt"] == admission["declaration"]["payload"]["source_text"]
          and candidate["operation_catalog_cid"] == semantic["operation_catalog_cid"], "public worker handoff changed complete signed task meaning")
    edit = candidate["edit"]
    _need(edit["path"] == "calc.py" and edit["effect"] == "modify"
          and base64.b64decode(edit["before_bytes_base64"], validate=True) == before
          and base64.b64decode(edit["after_bytes_base64"], validate=True) == after
          and edit["before_sha256"] == _sha(before) and edit["after_sha256"] == _sha(after),
          "public handoff bytes differ from the actual generated replacement")
    _need(bridge["generated_result_cid"] == generated["result_cid"]
          and bridge["reviewed_candidate_cid"] == generated["reviewed_candidate_cid"]
          and bridge["public_candidate_descriptor"] == descriptor
          and bridge["replacement_raw_cid"] == _cid(after)
          and bridge["before_raw_cid"] == _cid(before)
          and bridge["replacement_sha256"] == bridge["after_sha256"] == descriptor["after_sha256"] == _sha(after)
          and bridge["before_sha256"] == descriptor["before_sha256"] == _sha(before), "trusted proposal/public descriptor bridge differs")
    for field in ("task_cid", "task_revision", "finite_admission_cid", "semantic_context_cid"):
        _need(bridge[field] == descriptor[field] == candidate[field], "bridge changed native task/revision/admission identity")
    _need(bridge["task_cid"] in semantic["administrator_task_cids"] and descriptor["task_id"] == "FINITE-OFFSET",
          "bridge selected a foreign original native task")
    _need(bridge["advisory_context_cids"] == [contexts[label][0]["context_cid"] for label in
          ("initial_off", "root_frozen", "initial_child_frozen")], "bridge changed selected advisory context identities")
    _false(bridge, {"features_choose_fixed_edit", "execution_authority", "completion_authority", "publication_authority"}, "proposal bridge")
    _need(all(bridge[key] is True for key in ("canonical_source_unchanged", "native_task_rows_unchanged", "model_registry_unchanged")),
          "proposal bridge claims a changed owner")
    for name in ("generated_result_pin", "reviewed_candidate_pin", "replacement_artifact", "advisory_comparison_pin"):
        reader.pin(bridge[name])
    invariant = reader.json("native/proposal-owner-invariance.json")
    _need(invariant["task_rows_before"] == invariant["task_rows_after"]
          and invariant["registry_before"] == invariant["registry_after"], "trusted proposal changed actual owner rows")
    _need(set(invariant["source_before"]) == INVENTORY, "trusted proposal lost full canonical inventory")
    return bridge, descriptor


def audit(namespace):
    """Return an inert independent audit; read no path outside the copied tree."""
    reader = _Reader(namespace)
    report = reader.json("native/result.json")
    _need(report["schema"] == RESULT_SCHEMA and report["status"] == "completed", "joined native qualification did not complete")
    for name in ("worker_launched", "native_worker_successor_loop_qualified", "complete_task_population_retained",
                 "actual_public_checks_passed", "historical_artifacts_unchanged", "successor_cold_finite_outcomes_agree"):
        _need(report[name] is True, "joined qualification lacks actual " + name)
    _false(report, {"production_activated", "task_omission_authority", "universal_python_semantics_proved",
        "features_choose_fixed_edit", "latent_ranking_available", "formal_decoder_available", "cuda_qualified",
        "384d_qualified", "metadata_hydration_performed"}, "joined qualification")
    _need(report["provider_calls"] == report["training_steps_during_admission_and_worker"]
          == report["planning_model_calls"] == report["active_leases"] == report["waiting_requests"] == 0,
          "joined qualification claims provider/fitting work or leaked resources")
    _need(set(report["inventory_paths"]) == INVENTORY, "joined result does not account for ten source units")
    admission, initial, manifest, tasks = _admission(reader, "native/before-admission.json", offset=1, no_work=False)
    successor_admission, successor, successor_manifest, successor_tasks = _admission(
        reader, "native/successor-admission.json", offset=2, no_work=True)
    cold_admission, cold, _, _ = _admission(reader, "native/cold-admission.json", offset=2, no_work=True)
    _need(tasks == successor_tasks and manifest["tasks"] == successor_manifest["tasks"], "publication changed original signed task identities/meanings")
    _need(successor["head"]["generation"] == initial["head"]["generation"] + 1
          and successor["head"]["snapshot_cid"] != initial["head"]["snapshot_cid"], "successor is not a fresh source generation")
    _, initial_entries = _capture(reader, "native/initial-capture.json", initial["head"], final=False)
    _, successor_entries = _capture(reader, "native/successor-capture.json", successor["head"], final=True)
    _need(initial_entries["calc.py"]["source_cid"] == initial["source_cid"]
          and successor_entries["calc.py"]["source_cid"] == successor["source_cid"]
          and all(initial_entries[name]["source_cid"] == successor_entries[name]["source_cid"]
                  for name in INVENTORY - {"calc.py"}), "worker source changes exceeded actual requested symbol")
    roots = {"root_training": "root-training-context", "root_frozen": "root-frozen-context",
        "initial_child_training": "initial-child-training-context", "initial_child_frozen": "initial-child-frozen-context",
        "initial_off": "root-off-context", "successor_child_training": "successor-child-training-context",
        "successor_child_frozen": "successor-child-frozen-context", "successor_off": "successor-off-context"}
    contexts = {name: _context(reader, "native/private/advisory/" + suffix) for name, suffix in roots.items()}
    locators = report["audit_artifacts"]
    _need(locators["contexts"] == {name: "private/advisory/" + suffix for name, suffix in roots.items()},
          "audit context locator map differs from the closed retained layout")
    expected_previews = {name: "private/advisory/" + label + "-preview.json" for name, label in (
        ("initial_off", "initial-off"), ("root_frozen", "root-frozen"), ("initial_child_frozen", "initial-child-frozen"),
        ("successor_off", "successor-off"), ("successor_frozen", "successor-frozen"))}
    _need(locators["previews"] == expected_previews, "audit preview locator map differs from actual frozen modes")
    for name, (binding, _) in contexts.items():
        _need(binding["head"] == (successor["head"] if name.startswith("successor") else initial["head"]),
              "advisory mode belongs to a different current source generation")
    root, child, next_child = (contexts[name][0] for name in ("root_training", "initial_child_training", "successor_child_training"))
    _need(root["parent_version_id"] is None and child["parent_version_id"] == root["version_id"]
          and next_child["parent_version_id"] == child["version_id"], "three actual fitting phases are not exact ordered lineage")
    for frozen, training in (("root_frozen", "root_training"), ("initial_child_frozen", "initial_child_training"),
                             ("successor_child_frozen", "successor_child_training")):
        _need(contexts[frozen][0]["mode"] == "frozen" and contexts[frozen][1]["checkpoint"] == contexts[training][1]["checkpoint"],
              "selected frozen advice differs from genuine training checkpoint")
    initial_previews = {label: _preview(reader, "native/private/advisory/" + label + "-preview.json",
        contexts[context][0], initial, no_work=False) for label, context in
        (("initial-off", "initial_off"), ("root-frozen", "root_frozen"), ("initial-child-frozen", "initial_child_frozen"))}
    successor_previews = {label: _preview(reader, "native/private/advisory/" + label + "-preview.json",
        contexts[context][0], successor, no_work=True) for label, context in
        (("successor-off", "successor_off"), ("successor-frozen", "successor_child_frozen"))}
    initial_comparison = _comparison(reader, "native/initial-advisory-comparison.json", initial_previews)
    successor_comparison = _comparison(reader, "native/successor-advisory-comparison.json", successor_previews)
    total, phases, criteria = _training(reader, contexts, "native/training-attempts.json",
        "native/phase-costs.json", "native/criteria.json")
    _need(report["actual_attempted_training_epochs"] == total == 48, "whole experiment training accounting differs")
    _need(criteria["features_choose_fixed_edit"] is criteria["latent_ranking_available"] is criteria["formal_decoder_available"] is False,
          "predeclared scope grants unsupported learned authority")
    _need(criteria["training_selections"] == [["calc.py", "train"], ["known_variant.py", "train"],
          ["tune.py", "tune"], ["canary.py", "canary"]], "predeclared structural training split differs")
    ledger = reader.json("native/training-attempts.json")
    for row in ledger["attempts"]:
        binding, retained = _context(reader, row["context_path"])
        _need(binding["context_cid"] == row["context_cid"] and binding["version_id"] == row["version_id"]
              and binding["parent_version_id"] == row["parent_version_id"]
              and retained["invocation"]["operation_id"] == row["native_operation_id"],
              "actual fitting-attempt path/context/native operation binding differs")
    measurement = reader.json("native/training-measurements.json")
    _need(len(measurement["measurements"]) == 3 and len(measurement["continuations"]) == 2,
          "complete three-phase numerical measurement/two-continuation records missing")
    trained = {value[0]["version_id"]: value for value in contexts.values() if value[0]["mode"] == "train"}
    for row in measurement["measurements"]:
        binding, retained = trained[row["version_id"]]
        checkpoint = retained["checkpoint"]
        _need(row["context_cid"] == binding["context_cid"] and row["attempted_epochs"] == 16
              and row["selected_total_epochs"] == checkpoint["state"]["completed_epochs"]
              and row["before_tuning_objective"] == checkpoint["report"]["before"]["objective"]
              and row["after_tuning_objective"] == checkpoint["report"]["after"]["objective"]
              and row["tuning_nonincrease"] is True and row["all_recorded_losses_finite"] is True
              and math.isfinite(row["before_tuning_objective"]) and math.isfinite(row["after_tuning_objective"])
              and row["after_tuning_objective"] <= row["before_tuning_objective"], "numerical training measurement does not replay")
    _need({(row["parent_version_id"], row["child_version_id"]) for row in measurement["continuations"]}
          == {(root["version_id"], child["version_id"]), (child["version_id"], next_child["version_id"])},
          "numerical continuation records name a different actual lineage")
    bridge, descriptor = _proposal(reader, admission, initial, contexts)
    lifecycle, scope = _worker(reader, report, descriptor["task_cid"])
    _need(scope["candidate"]["descriptor"] == descriptor and scope["head"] == initial["head"]
          and scope["finite_admission_cid"] == _structured_cid(admission), "signed execution scope lost selected public handoff")
    population = scope["native_population"]
    type_cid = tasks["FINITE-TYPE"]["content_id"]
    _need(set(population["completed_prerequisites"]) == {type_cid}
          and set(population["completion_rows"]) == {type_cid}, "native completed prerequisite evidence is missing")
    _need(population["selected_task_cids"] == [descriptor["task_cid"]], "native launch admitted a different residual")
    checks = reader.json("native/prerequisite-validation.json")
    _need(checks["passed"] is True and checks["task_cid"] == type_cid, "genuine prerequisite public check did not pass")
    materialized = reader.json("native/materialized.json")
    _need(set(materialized["task_cids"]) == set(initial["administrator_task_cids"])
          and materialized["administrator_task_population_preserved"] is True, "native materialization omitted original tasks")
    reader.pin(materialized["finite_admission_ref"])
    _need(reader.json(materialized["finite_admission_ref"]["path"]) == admission, "native plan raw reference differs from full signed admission")
    no_fit = reader.json("native/worker-no-fit-observation.json")
    _need(no_fit["registry_before"] == no_fit["registry_after"] and no_fit["unchanged"] is True
          and no_fit["training_steps_during_admission_and_worker"] == no_fit["planning_model_calls"] == 0,
          "admission/native worker changed numerical registry")
    _need(len(locators["no_fit_inventories"]) == 8, "complete five-preview/admission/worker no-fit inventories missing")
    for path in locators["no_fit_inventories"]:
        value = reader.json("native/" + path)
        before_inventory = value.get("before", value.get("registry_before"))
        after_inventory = value.get("after", value.get("registry_after"))
        _need(type(before_inventory) is dict and before_inventory == after_inventory and value["unchanged"] is True,
              "actual advisory/admission no-fit inventory changed")
    stale = reader.json("native/stale-advisory-refusals.json")
    _need({row["label"] for row in stale} == {"root-frozen", "initial-child-frozen"}
          and all(row["rejected"] is True and row["error"] for row in stale), "old-source advice current refusals missing")
    cold_comparison = reader.json("native/cold-comparison.json")
    _need(cold_comparison["agreement"] is True and set(cold_comparison["compared_fields"]) == COLD_FIELDS
          and all(successor[name] == cold[name] for name in COLD_FIELDS), "independent cold nine-field finite replay disagrees")
    _need(cold["head"]["generation"] == 1, "independent cold finite catalog was replaced by original successor")
    request = reader.json("native/cold-request.json")
    fresh = reader.json("native/fresh-process-replay.json")
    _need(locators["fresh_process_response"] == "fresh-process-response.json",
          "fresh replay response locator differs from the closed protocol artifact")
    response = reader.json("native/fresh-process-response.json")
    _need(response == fresh, "fresh replay response artifact differs from retained replay result")
    _need(fresh["schema"] == "finite-advisory-worker-current-and-historical-replay@1"
          and fresh["historical_integrity_verified"] is True and fresh["current_freshness_claimed"] is False
          and fresh["task_statuses"] == {"FINITE-TYPE": "completed", "FINITE-OFFSET": "completed"},
          "fresh same-catalog replay lost original complete historical task population")
    _need(fresh["source_head"] == request["head"] == successor["head"]
          and fresh["selected_context_binding"] == request["selected_context_binding"] == contexts["successor_child_frozen"][0],
          "fresh selected successor advice differs from original successor catalog")
    _false(fresh, {"fitting_performed", "promotion_performed", "independent_cold_generation_used_for_feature_verification"},
           "fresh current successor replay")
    _need(fresh["training_steps"] == 0 and fresh["current_successor_feature_verified"] is True,
          "fresh numerical replay fitted or did not verify current selected child")
    parent = request["registry_before_reopen"]
    before, after = fresh["registry_before_verification"], fresh["registry_after_verification"]
    _need(before == after and before["owner_generation"] == parent["owner_generation"] + 1
          and before["artifacts"] == parent["artifacts"]
          and {key: value for key, value in before["tables"].items() if key != "meta"}
          == {key: value for key, value in parent["tables"].items() if key != "meta"},
          "fresh registry changed beyond legitimate owner generation or verification mutated rows")
    for pin in request["retained_pins"]:
        reader.pin(pin)
    for pin in reader.json("native/historical-parent-artifact-pins.json"):
        reader.pin(pin)
    for pin in report["execution_sources"]:
        reader.pin(pin)
    snapshots = reader.json("selected-source-snapshot.json")
    source_count = 0
    for family, rows in snapshots.items():
        _need(family in {"source", "datasets", "kit"}, "unexpected copied source family")
        for pin in rows:
            raw = reader.raw(family + "/" + pin["relative"])
            _need(len(raw) == pin["bytes"] and _sha(raw) == pin["sha256"], "selected readonly source snapshot changed")
            source_count += 1
    container = reader.json("container-execution.json")
    _need(container["returncode"] == 0 and container["container_removed"] is True
          and container["container_results_copied"] is True and container["candidate_handoffs_copied"] is True,
          "bounded container execution/copy/removal was incomplete")
    reader.unchanged()
    return {"schema": SCHEMA, "status": "verified", "archive_integrity_and_join_verified": True,
        "advisory_fields": list(ADVISORY_FIELDS), "independent_cold_finite_fields": sorted(COLD_FIELDS),
        "source_units": len(INVENTORY), "original_tasks": len(tasks), "finite_inputs": 5,
        "actual_attempted_epochs": total, "selected_initial_model": child["version_id"],
        "selected_successor_model": next_child["version_id"], "phase_costs": phases,
        "readonly_source_files_checked": source_count, "retained_files_checked": len(reader.observed),
        "retained_bytes_unchanged_during_audit": True, "worker_task_cid": descriptor["task_cid"],
        "worker_preclaim_revision": descriptor["task_revision"], "initial_mode_contexts_distinct": True,
        "source_model_successor_generation": successor["head"]["generation"],
        "fitting_performed": False, "inference_performed": False, "checkers_rerun": False,
        "metadata_hydration_performed": False, "proof_authority": False, "execution_authority": False,
        "universal_python_semantics_verified": False, "process_origin_attested": False,
        "scope": "Copied retained artifact integrity, public-key historical signatures and exact bounded joins; no current authority grant or transitive environment attestation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.namespace)
    except (AdvisoryWorkerAuditError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "refused", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
