"""A pinned finite-context edit in an allocated native worker worktree.

The author checks the original admission and actual native rows. The worker
checks public signed context and exact source preimages without owner keys.
Neither operation claims a worker claim, theorem, publication or completion;
the native dispatch, acceptance and owner completion paths retain those gates.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import stat
import sys

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)

from ..control.profile_authority import verify_did_key_signature
from .doctor_candidate_runner import MAX_BYTES, _directory, _git, _read, _sha, _unique
from .doctor_contract_candidate_runner import _check_parent, _path, _write

SCHEMA = "supervisor-finite-repository-candidate@1"
SCOPE = "one_closed_integer_offset_edit_in_allocated_native_worktree"
_FALSE = {name: False for name in (
    "source_semantics_verified", "proof_authority", "execution_authority",
    "publication_authority", "completion_authority", "task_omission_authority",
)}
FIELDS = frozenset({
    "schema", "repository", "baseline_commit", "task_cid", "task_id", "task_revision",
    "finite_admission", "finite_admission_cid", "semantic_context_cid", "original_prompt",
    "original_clause_ids", "operation_catalog_cid", "permitted_outputs", "edit",
    "provider_calls", "training_steps", "scope", "candidate_cid", *_FALSE,
})
_EDIT_FIELDS = {"path", "effect", "before_sha256", "after_sha256",
                "before_bytes_base64", "after_bytes_base64"}
_CONTEXT_FALSE = (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "code_proof_authority", "production_admitted", "production_activation",
    "execution_authority", "completion_authority", "mutation_authority",
    "omission_authority", "worker_launched", "convergence_proved",
)


def _need(value, message):
    if not value:
        raise ValueError(message)


def _same(left, right):
    return canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)


def _plain(value):
    raw = canonical_dag_json_bytes(value)
    _need(len(raw) <= MAX_BYTES, "finite candidate exceeds its byte bound")
    return json.loads(raw, object_pairs_hook=_unique)


def _public(envelope, binding=None, *, mirror=True):
    _need(type(envelope) is dict and set(envelope) == {"payload", "binding"},
          "closed public signed finite envelope required")
    observed = envelope["binding"]
    _need(type(observed) is dict and set(observed) == {"identity", "profile_id", "signature"}
          and all(type(observed[key]) is str and observed[key] for key in observed),
          "exact public finite signer identity required")
    if binding is not None:
        _need(all(observed[key] == binding[key] for key in ("identity", "profile_id")),
              "finite context belongs to a different signer")
    verify_did_key_signature(identity_did=observed["identity"], payload=envelope["payload"],
                             signature=observed["signature"], mirror=mirror)
    return envelope["payload"]


def _decode_source(edit, name):
    encoded = edit[name + "_bytes_base64"]
    _need(type(encoded) is str and len(encoded) <= 90_000, "bounded literal candidate source required")
    raw = base64.b64decode(encoded, validate=True)
    _need(0 < len(raw) <= 65_536 and base64.b64encode(raw).decode("ascii") == encoded
          and type(edit[name + "_sha256"]) is str
          and _sha(raw) == edit[name + "_sha256"], "candidate source encoding or digest differs")
    return raw


def _context(candidate, *, mirror=True):
    """Public integrity/context closure, never current owner authorization."""
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
        IntegerOffsetContract, compile_integer_offset,
    )
    from ..planning import finite_integer_codebase as matcher
    from ..prompt.prompt_workflow import PromptGoalGraph

    admission = candidate["finite_admission"]
    _need(type(admission) is dict and set(admission) ==
          {"declaration", "graph", "evidence", "local_admission", "receipt"}
          and candidate["finite_admission_cid"] == cid_for_structured(admission),
          "complete exact original finite admission required")
    declaration = _public(admission["declaration"], mirror=mirror)
    signer = admission["declaration"]["binding"]
    manifest_envelope = declaration["manifest"]
    manifest = _public(manifest_envelope, signer, mirror=mirror)
    receipt = _public(admission["receipt"], signer, mirror=mirror)
    old = admission["local_admission"]
    _need(type(old) is dict and set(old) == {"manifest", "graph", "receipt"}
          and _same(old["manifest"], manifest_envelope) and _same(old["graph"], admission["graph"]),
          "original guarded local admission required")
    _public(old["receipt"], signer, mirror=mirror)
    semantic = receipt["semantic_context"]
    _need(declaration.get("schema") == "supervisor-finite-repository-declaration@1"
          and declaration.get("profile") == "finite-repository-fixed-administrator-population@1"
          and receipt.get("schema") == "supervisor-finite-repository-admission@1"
          and receipt.get("profile") == declaration["profile"]
          and receipt.get("planning_permitted") is True and receipt.get("no_work_review_only") is False
          and all(receipt.get(name) is False and semantic.get(name) is False for name in _CONTEXT_FALSE)
          and receipt["declaration_cid"] == cid_for_structured(admission["declaration"])
          and receipt["graph_cid"] == cid_for_structured(admission["graph"])
          and receipt["evidence_cid"] == cid_for_structured(admission["evidence"])
          and receipt["local_admission_cid"] == cid_for_structured(old)
          and receipt["semantic_context_cid"] == cid_for_structured(semantic)
          and candidate["semantic_context_cid"] == receipt["semantic_context_cid"],
          "public finite receipt or fact authority differs")
    query = matcher.prepare_finite_integer_query(intent_document=decode_intent_ir(declaration["intent_json"]),
                                               source_text=declaration["source_text"])
    _need(query.get("supported") is True
          and _same(query, semantic["query"]) and _same(query, admission["evidence"]["match"]["query"])
          and candidate["original_prompt"] == declaration["source_text"]
          and _same(candidate["original_clause_ids"], query["requirement_ids"])
          and _same(semantic["eligible_requirement_ids"], [matcher.TYPE_STATEMENT_ID])
          and _same(semantic["residual_requirement_ids"], [matcher.OFFSET_STATEMENT_ID])
          and candidate["operation_catalog_cid"] == semantic["operation_catalog_cid"]
          and candidate["operation_catalog_cid"] == cid_for_structured(declaration["operation_catalog"]),
          "complete original finite instruction, catalog or clause partition differs")
    graph = PromptGoalGraph.from_dict(admission["graph"])
    bindings = semantic["native_task_bindings"]
    _need(len(graph.tasks) == 2 and set(bindings) == {matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID}
          and set(declaration["task_bindings"]) == set(bindings)
          and _same(semantic["administrator_task_cids"], sorted(task.task_cid for task in graph.tasks)),
          "complete original two-task population required")
    by_cid = {task.task_cid: task for task in graph.tasks}
    for clause, bound in bindings.items():
        _need(type(bound) is dict and set(bound) == {"task_key", "task_cid"}
              and bound["task_cid"] in by_cid and by_cid[bound["task_cid"]].task_key == bound["task_key"]
              and declaration["task_bindings"][clause] == bound["task_key"],
              "original finite clause-to-task binding differs")
    residual = bindings[matcher.OFFSET_STATEMENT_ID]
    prerequisite = bindings[matcher.TYPE_STATEMENT_ID]
    task = by_cid[residual["task_cid"]]
    _need(candidate["task_cid"] == residual["task_cid"] and candidate["task_id"] == residual["task_key"]
          and task.dependency_task_cids == (prerequisite["task_cid"],)
          and candidate["repository"] == manifest["repository"]
          and candidate["baseline_commit"] == manifest["baseline_commit"],
          "candidate differs from the exact residual native task or baseline")
    spec = next(row for row in manifest["tasks"] if row["task_key"] == task.task_key)
    contract = IntegerOffsetContract.from_dict(query["contract"])
    _need(len(spec["outputs"]) == 1 and spec["outputs"][0]["path"] == contract.path
          and spec["outputs"][0]["effect"] == "modify"
          and _same(candidate["permitted_outputs"], spec["outputs"]),
          "one exact signed integer source output required")
    edit = candidate["edit"]
    _need(type(edit) is dict and set(edit) == _EDIT_FIELDS and edit["path"] == contract.path
          and edit["effect"] == "modify", "closed one-output finite edit required")
    _path(edit["path"])
    before, after = _decode_source(edit, "before"), _decode_source(edit, "after")
    source = admission["evidence"]["match"]["observation"]["artifacts"]["source"]
    _need(cid_for_bytes(before) == semantic["source_cid"] == source["cid"]
          and edit["before_sha256"] == source["sha256"] and len(before) == source["size_bytes"],
          "finite candidate preimage differs from the original observation")
    previous = compile_integer_offset(before, contract, revision="candidate:before", mirror=mirror)
    proposed = compile_integer_offset(after, contract, revision="candidate:after", mirror=mirror)
    _need(previous.body_offset != contract.offset and proposed.body_offset == contract.offset,
          "candidate must repair the residual in the closed integer profile")
    return admission, manifest, graph, semantic, before, after


def _validate(candidate, *, mirror=True):
    _need(type(candidate) is dict and set(candidate) == FIELDS and candidate["schema"] == SCHEMA
          and candidate["scope"] == SCOPE
          and all(candidate[name] is False for name in _FALSE)
          and type(candidate["task_revision"]) is int and candidate["task_revision"] >= 1
          and type(candidate["provider_calls"]) is int and candidate["provider_calls"] == 0
          and type(candidate["training_steps"]) is int and candidate["training_steps"] == 0
          and type(candidate["baseline_commit"]) is str
          and re.fullmatch(r"[0-9a-f]{40}", candidate["baseline_commit"]) is not None
          and candidate["candidate_cid"] == cid_for_structured(
              {key: value for key, value in candidate.items() if key != "candidate_cid"}),
          "closed finite candidate identity, revision or authority differs")
    return _context(candidate, mirror=mirror)


def load_finite_repository_candidate(*, artifact: Path, expected_sha256: str, mirror: bool = True) -> dict:
    """Read a pinned, readonly external handoff; no private keys or fitting."""
    path = Path(artifact).absolute()
    _need(path.resolve(strict=True) == path, "exact non-symlink finite artifact required")
    parent = _directory(path.parent)
    try:
        raw, info = _read(parent, path.name)
        directory = os.fstat(parent)
        _need(not directory.st_mode & 0o022, "finite artifact directory must forbid other writers")
    finally:
        os.close(parent)
    _need(not stat.S_IMODE(info.st_mode) & 0o222 and _sha(raw) == expected_sha256,
          "readonly pinned finite artifact required")
    candidate = json.loads(raw, object_pairs_hook=_unique)
    _validate(candidate, mirror=mirror)
    canonical = Path(candidate["repository"])
    _need(canonical.is_absolute() and canonical.resolve(strict=True) == canonical
          and not path.is_relative_to(canonical), "finite artifact must remain outside canonical source")
    return candidate


def _native_rows(intent, old, verified, semantic):
    from . import local_planning_admission as local
    rows = [local._plain(dict(row)) for row in intent.list_tasks(limit=17)]
    expected = {task.task_cid: task for task in verified["graph"].tasks}
    _need({row["task_cid"] for row in rows} == set(expected) and len(rows) == 2,
          "native owner population differs from the original two-task admission")
    for row in rows:
        envelope = row["body"].get(local.CONTRACT_KEY)
        contract = local._verify_signature(envelope, verified["profile"])
        owner_id = contract.get("intent_owner_id")
        _need(type(owner_id) is str and owner_id, "original native contract owner required")
        rebuilt = local._pending_contract_payload(admission=old, verified=verified,
            task=expected[row["task_cid"]], intent_owner_id=owner_id)
        _need(_same(contract, rebuilt) and row["task_alias"] == rebuilt["task_key"]
              and row["identity"].get("local_contract_cid") == local.content_identity(envelope)
              and _same(row["dependencies"], rebuilt["dependencies"])
              and row["plan_cid"] == rebuilt["plan_id"]
              and row["goal_cid"] == expected[row["task_cid"]].goal_cid,
              "original native pending completion guard differs")
    bindings = semantic["native_task_bindings"]
    from ..planning.finite_integer_codebase import TYPE_STATEMENT_ID, OFFSET_STATEMENT_ID
    by_cid = {row["task_cid"]: row for row in rows}
    first = by_cid[bindings[TYPE_STATEMENT_ID]["task_cid"]]
    selected = by_cid[bindings[OFFSET_STATEMENT_ID]["task_cid"]]
    _need(first["status"] == "completed" and selected["status"] == "ready"
          and type(selected["revision"]) is int and selected["revision"] >= 1,
          "actual completed type prerequisite and ready residual required")
    with intent._connection(write=False) as connection:
        _need(type(first["revision"]) is int and first["revision"] >= 2
              and not local.local_completion_missing(connection, first["task_cid"],
                                                     first["body"], first["revision"] - 1),
              "completed type prerequisite lacks actual current public-check evidence")
    return rows, selected


def _source_fence(manifest):
    """Detached source reads after executable verification callbacks finish."""
    root = Path(manifest["repository"])
    for name, expected in sorted(manifest["sources"].items()):
        relative = _path(name)
        parent = _directory(root / relative.parent)
        try:
            raw, metadata = _read(parent, relative.name)
            _check_parent(root, relative, parent)
            _need(_sha(raw) == expected["sha256"]
                  and bool(metadata.st_mode & 0o111) is expected["executable"],
                  "original canonical source changed before author return")
        finally:
            os.close(parent)


def author_finite_repository_candidate(*, admission, intent, task_cid: str,
        after_bytes: bytes, output: Path) -> dict:
    """Derive a readonly handoff from verified context and genuine native rows.

    ``task_revision`` is the observed ready revision, not a predicted claim
    revision. Native dispatch must independently check it and bind its real
    admitted attempt/claim/fence. Bound owner read views are supported.
    """
    from ..task_sources.intent_repository import IntentRepository
    from . import finite_repository_admission as boundary
    from . import local_planning_admission as local
    _need(type(intent) is IntentRepository and type(after_bytes) is bytes,
          "native intent owner and exact candidate bytes required")
    verified = boundary.verify_finite_repository_admission(admission=admission)
    frozen, semantic = verified["admission"], verified["semantic_context"]
    _need(frozen["local_admission"] is not None, "no-work context cannot author a worker candidate")
    old = frozen["local_admission"]
    native = local.verify_local_benchmark_admission(old, initial=True)
    rows, selected = _native_rows(intent, old, native, semantic)
    _need(selected["task_cid"] == task_cid, "candidate task is not the original residual")
    declaration = frozen["declaration"]["payload"]
    source = frozen["evidence"]["match"]["observation"]["artifacts"]["source"]
    parent = _directory(Path(source["path"]).parent)
    try:
        before, _ = _read(parent, Path(source["path"]).name)
    finally:
        os.close(parent)
    manifest = native["manifest"]
    spec = next(row for row in manifest["tasks"] if row["task_key"] == selected["task_alias"])
    candidate = {"schema": SCHEMA, "scope": SCOPE, "repository": manifest["repository"],
        "baseline_commit": manifest["baseline_commit"], "task_cid": task_cid,
        "task_id": selected["task_alias"], "task_revision": selected["revision"],
        "finite_admission": frozen, "finite_admission_cid": cid_for_structured(frozen),
        "semantic_context_cid": cid_for_structured(semantic), "original_prompt": declaration["source_text"],
        "original_clause_ids": semantic["query"]["requirement_ids"],
        "operation_catalog_cid": semantic["operation_catalog_cid"], "permitted_outputs": spec["outputs"],
        "edit": {"path": semantic["query"]["contract"]["path"], "effect": "modify",
            "before_sha256": _sha(before), "after_sha256": _sha(after_bytes),
            "before_bytes_base64": base64.b64encode(before).decode("ascii"),
            "after_bytes_base64": base64.b64encode(after_bytes).decode("ascii")},
        "provider_calls": 0, "training_steps": 0, **_FALSE}
    candidate["candidate_cid"] = cid_for_structured(candidate)
    candidate = _plain(candidate)
    _validate(candidate)
    path = Path(output).absolute()
    repository = Path(manifest["repository"])
    _need(path.resolve(strict=False) == path and not path.exists()
          and not path.is_relative_to(repository), "fresh external finite candidate artifact required")
    parent = _directory(path.parent)
    try:
        info = os.fstat(parent)
        _need(info.st_uid == os.geteuid() and not info.st_mode & 0o022,
              "owner-controlled finite artifact directory required")
        raw = canonical_dag_json_bytes(candidate)
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.fsync(parent)
        _need(_read(parent, path.name)[0] == raw, "retained finite candidate bytes differ")
    finally:
        os.close(parent)
    boundary.verify_finite_repository_admission(admission=frozen)
    final_native = local.verify_local_benchmark_admission(old, initial=True)
    final_rows, _ = _native_rows(intent, old, final_native, semantic)
    _need(_same(rows, final_rows), "native task generation changed while authoring candidate")
    _need(_same(load_finite_repository_candidate(artifact=path, expected_sha256=_sha(raw)), candidate),
          "finite candidate changed before author return")
    _need(_git(repository, "rev-parse", "HEAD").decode().strip() == manifest["baseline_commit"],
          "canonical baseline changed before author return")
    final_rows = [local._plain(dict(row)) for row in intent.list_tasks(limit=17)]
    _need(_same(final_rows, rows), "native ready generation changed before author return")
    _source_fence(manifest)
    parent = _directory(path.parent)
    try:
        final_raw, final_info = _read(parent, path.name)
        _need(final_raw == raw and not final_info.st_mode & 0o222,
              "readonly finite candidate changed before author return")
    finally:
        os.close(parent)
    return {"artifact": str(path), "sha256": _sha(raw), "candidate_cid": candidate["candidate_cid"],
        "finite_admission_cid": candidate["finite_admission_cid"],
        "semantic_context_cid": candidate["semantic_context_cid"], "task_cid": task_cid,
        "task_id": candidate["task_id"], "task_revision": candidate["task_revision"],
        "before_sha256": candidate["edit"]["before_sha256"],
        "after_sha256": candidate["edit"]["after_sha256"]}


def materialize_finite_repository_candidate(*, artifact: Path, expected_sha256: str,
        task_cid: str, prompt: str, workspace: Path, mirror: bool = True) -> dict:
    """Write one exact source candidate; commits and acceptance stay native."""
    candidate = load_finite_repository_candidate(artifact=artifact, expected_sha256=expected_sha256, mirror=mirror)
    _need(candidate["task_cid"] == task_cid and type(prompt) is str and len(prompt.encode()) <= 256_000,
          "bounded exact native task prompt required")
    wire, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    _need(type(wire) is dict and wire.get("objective_id") == candidate["task_id"],
          "native objective differs from the finite residual task")
    root, canonical = Path(workspace).absolute(), Path(candidate["repository"])
    artifact = Path(artifact).absolute()
    _need(root.resolve(strict=True) == root and root != canonical
          and not root.is_relative_to(canonical) and not canonical.is_relative_to(root)
          and not artifact.is_relative_to(root),
          "separate allocated native worktree and external artifact required")
    _need(Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()) == root
          and _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir") ==
              _git(canonical, "rev-parse", "--path-format=absolute", "--git-common-dir"),
          "finite candidate belongs to a foreign Git repository")
    baseline = candidate["baseline_commit"]
    _need(_git(root, "rev-parse", "HEAD").decode().strip() == baseline
          and _git(canonical, "rev-parse", "HEAD").decode().strip() == baseline,
          "canonical or allocated baseline drifted")
    edit = candidate["edit"]
    relative = _path(edit["path"])
    before, after = _decode_source(edit, "before"), _decode_source(edit, "after")
    _need(_git(root, "show", baseline + ":" + str(relative)) == before,
          "finite preimage differs from Git baseline")
    parent, canonical_parent = _directory(root / relative.parent), _directory(canonical / relative.parent)
    try:
        current, metadata = _read(parent, relative.name)
        _need(current == before and _read(canonical_parent, relative.name)[0] == before,
              "allocated or canonical finite source preimage drifted")
        _check_parent(root, relative, parent)
        _check_parent(canonical, relative, canonical_parent)
        # Re-read the complete pinned handoff immediately before source writes.
        _need(_same(load_finite_repository_candidate(artifact=artifact, expected_sha256=expected_sha256, mirror=mirror), candidate),
              "finite handoff changed before materialization")
        mode = _write(parent, relative, after, metadata)
        _need(_same(load_finite_repository_candidate(artifact=artifact, expected_sha256=expected_sha256, mirror=mirror), candidate),
              "finite handoff changed before final receipt")
        # Complete executable/native verification before detached descriptor
        # reads. A late callback must not mutate source and escape the fence.
        _need(_git(root, "rev-parse", "HEAD").decode().strip() == baseline
              and _git(canonical, "rev-parse", "HEAD").decode().strip() == baseline,
              "finite baseline changed during materialization")
        _check_parent(root, relative, parent)
        _check_parent(canonical, relative, canonical_parent)
        _need(_read(parent, relative.name)[0] == after
              and _read(canonical_parent, relative.name)[0] == before,
              "finite source changed during materialization")
    finally:
        os.close(parent)
        os.close(canonical_parent)
    return {"schema": "native-finite-repository-candidate-materialization@1", "status": "candidate_materialized",
        "candidate_cid": candidate["candidate_cid"], "finite_admission_cid": candidate["finite_admission_cid"],
        "semantic_context_cid": candidate["semantic_context_cid"], "task_cid": task_cid,
        "task_id": candidate["task_id"], "ready_task_revision": candidate["task_revision"],
        "baseline_commit": baseline, "writes": [{"path": str(relative), "effect": "modify",
            "after_sha256": edit["after_sha256"], "write_mode": mode}],
        "provider_calls": 0, "training_steps": 0, **_FALSE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task-cid", required=True)
    args = parser.parse_args()
    try:
        result = materialize_finite_repository_candidate(artifact=args.artifact,
            expected_sha256=args.sha256, task_cid=args.task_cid,
            prompt=sys.stdin.buffer.read(256_001).decode(), workspace=Path.cwd())
    except Exception as error:
        print(json.dumps({"schema": "native-finite-repository-candidate-materialization@1",
            "status": "refused", "error_type": type(error).__name__, **_FALSE}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "SCOPE", "FIELDS", "author_finite_repository_candidate",
           "load_finite_repository_candidate", "materialize_finite_repository_candidate"]
