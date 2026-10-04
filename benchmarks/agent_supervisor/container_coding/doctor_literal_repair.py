"""Closed missing-export repair using pinned AST literals and native Doctor gates.

History nominates values; the current caller and existing fixture independently
constrain them. This scoped operator is not general autonomous synthesis.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import math
from pathlib import Path
import re
import subprocess
import sys

from doctor_symbol_repair import apply_candidate, digest, save

TARGET = "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
CALLER = "ipfs_accelerate_py/agent_supervisor/todo_daemon/retained_callback_suffix.py"
FIXTURE = "test/api/test_retained_callback_cooldown.py"
NAMES = (
    "DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2",
    "_DATABASE_POST_MERGE_CALLBACK_INTEGRATION_RECOVERY_RECEIPT_FIELDS",
)


def literal_exports(source, names):
    """Interpret only bounded immutable declarations, without executing source."""
    if len(source.encode()) > 8_000_000:
        raise ValueError("donor exceeds source bound")
    tree = ast.parse(source)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions.setdefault(target.id, []).append(node.value)
    active = set()

    def resolve(name):
        if name in active or len(active) >= 32 or len(definitions.get(name, [])) != 1:
            raise ValueError("cyclic, missing or ambiguous literal: " + name)
        active.add(name)
        try:
            return value(definitions[name][0])
        finally:
            active.remove(name)

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return resolve(node.id)
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            result = []
            for item in node.elts:
                if isinstance(item, ast.Starred):
                    expanded = value(item.value)
                    if not isinstance(expanded, frozenset):
                        raise ValueError("starred value must be immutable set")
                    result.extend(expanded)
                else:
                    result.append(value(item))
            if len(result) > 256 or not all(isinstance(x, str) for x in result):
                raise ValueError("literal set exceeds contract")
            return frozenset(result)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "frozenset"
            and len(node.args) == 1
            and not node.keywords
        ):
            result = value(node.args[0])
            if isinstance(result, frozenset):
                return result
        raise ValueError("executable or unsupported donor expression")

    return {name: resolve(name) for name in names}


def propose(before, caller, donor):
    tree = ast.parse(before)
    # Conservatively reject every existing binding, including nested bindings.
    bound = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    bound.update(
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    bound.update(
        n.asname or n.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.alias)
    )
    imports = {
        a.name
        for n in ast.walk(ast.parse(caller))
        if isinstance(n, ast.ImportFrom) and n.module == "implementation_daemon"
        for a in n.names
    }
    if set(NAMES) & bound or not set(NAMES) <= imports:
        raise ValueError("exports already bound or current caller does not require them")
    values = literal_exports(donor, NAMES)
    lines = []
    for name, value in values.items():
        expression = (
            repr(value) if isinstance(value, str) else "frozenset(" + repr(sorted(value)) + ")"
        )
        lines.append(name + " = " + expression)
    declarations = "\n\n".join(lines) + "\n"
    after = (
        before
        + b"\n\n# Retained-callback recovery contracts restored from pinned history.\n"
        + declarations.encode()
    )
    updated = ast.parse(after)
    if ast.dump(ast.Module(body=updated.body[: -len(NAMES)], type_ignores=[])) != ast.dump(tree):
        raise ValueError("repair changed existing syntax")
    return after, values, declarations


def run(root, output, revision):
    import json
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import (
        build_program_evidence_index,
        build_language_edge_program_graph,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        build_code_symbol_vector_index,
        search_code_symbol_vector_index,
        CodeVectorIndexSnapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
        DoctorAuthorityRoots,
        DeterministicDoctorFinding,
        DoctorRepairDisposition,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician import (
        DeterministicDoctorTactician,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
        SupervisorMetaIndex,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
        ProgramWorldDatabase,
    )

    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(
        ["git", "rev-parse", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    donor = subprocess.check_output(["git", "show", commit + ":" + TARGET], cwd=root).decode()
    before = (root / TARGET).read_bytes()
    caller = (root / CALLER).read_text()
    fixture_source = (root / FIXTURE).read_bytes()
    after, values, declarations = propose(before, caller, donor)
    # Existing fixture is independently authored current contract evidence. It is
    # executed locally, never donor code; exact bytes are bound and rechecked.
    sys.path.insert(0, str(root))
    from test.api.test_retained_callback_cooldown import fixture, ROUTE

    task, _, _ = fixture()
    predecessor = task["body"]["completion_receipt"]
    expected = {
        NAMES[0]: predecessor["post_merge_completion_recovery_seed"]["schema"],
        NAMES[1]: frozenset(predecessor) - ROUTE - {"backoff_ms", "retry_not_before_ms"},
    }
    if values != expected:
        raise ValueError("historical literals disagree with independent fixture contract")
    bindings = {
        "before": digest(before),
        "after": digest(after),
        "caller": digest(caller.encode()),
        "fixture": digest(fixture_source),
        "donor": digest(donor.encode()),
        "commit": commit,
    }
    cid = content_identity(bindings)
    save(
        output / "proposal.json",
        {
            "operator": "closed-immutable-export-restoration@1",
            "bindings": bindings,
            "symbols": NAMES,
            "unrelated_ast_unchanged": True,
            "donor_code_executed": False,
            "scope_selected_by": "assistant",
            "candidate_values_selected_by": "bounded AST operator",
        },
    )
    # Fragment paths are explicit: this is not a whole-repository AST snapshot.
    documents = {"history/declarations.py": declarations, CALLER: caller}
    ast_index = build_program_evidence_index(documents)
    graph = build_language_edge_program_graph(ast_index, forest_id=cid)
    save(output / "ast.json", ast_index.ast_index.to_dict())
    save(output / "graph.json", graph.to_dict())

    # Transparent lexical vectors. No learned embedding provider is claimed.
    def tokens(text):
        return re.findall("[a-z0-9]+", text.lower())

    terms = [Counter(tokens(text)) for text in documents.values()]
    vocabulary = sorted(set().union(*(set(t) for t in terms)))
    idf = {
        t: 1 + math.log((1 + len(terms)) / (1 + sum(t in doc for doc in terms))) for t in vocabulary
    }

    def vector(text):
        counts = Counter(tokens(text))
        values = [counts[t] * idf[t] for t in vocabulary]
        norm = math.sqrt(sum(v * v for v in values))
        if not norm:
            raise ValueError("query has no corpus terms")
        return [v / norm for v in values]

    consumer_node = next(
        n
        for n in ast.parse(caller).body
        if isinstance(n, ast.FunctionDef) and n.name == "verified_seed_predecessor"
    )
    consumer_fragment = ast.get_source_segment(caller, consumer_node)
    vector_ast = build_program_evidence_index(
        {"fragments/verified_seed_predecessor.py": consumer_fragment}
    )
    save(
        output / "vector-scope.json",
        {
            "scope": "one actual consumer function fragment",
            "source_sha256": digest(caller.encode()),
            "line_start": consumer_node.lineno,
            "line_end": consumer_node.end_lineno,
            "limitation": "full caller vector indexing fails: native effect reference exceeds 320 bytes; historical constants are not native qualified symbols",
        },
    )
    vector_index = build_code_symbol_vector_index(
        vector_ast.ast_index,
        forest_id=cid,
        tree_id=cid,
        model_id="lexical-tfidf-symbols@1",
        dimensions=len(vocabulary),
        configuration_id=content_identity(
            {"vocabulary": vocabulary, "idf_hex": {k: v.hex() for k, v in idf.items()}}
        ),
        vectors=lambda row: vector(row.qualified_symbol),
    )
    # Persist and reopen the native index through an actual DuckDB catalog.
    db = output / "vectors.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE snapshots (id VARCHAR PRIMARY KEY, body JSON)")
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?)",
            [vector_index.index_id, json.dumps(vector_index.to_dict())],
        )
    with duckdb.connect(str(db), read_only=True) as conn:
        hydrated = CodeVectorIndexSnapshot.from_dict(
            json.loads(
                conn.execute(
                    "SELECT body FROM snapshots WHERE id=?", [vector_index.index_id]
                ).fetchone()[0]
            )
        )
    hits = {
        name: search_code_symbol_vector_index(hydrated, vector(name), max_results=5).to_dict()
        for name in NAMES
    }
    save(output / "vector-query.json", hits)
    save(
        output / "vector-model.json",
        {
            "kind": "lexical TF-IDF; not learned semantic embeddings",
            "vocabulary": vocabulary,
            "idf": idf,
        },
    )
    roots = DoctorAuthorityRoots(
        **{name: cid for name in DoctorAuthorityRoots.__dataclass_fields__ if name != "SCHEMA"}
    )
    plans = []
    for name in NAMES:
        finding = DeterministicDoctorFinding(
            roots=roots,
            finding_id=content_identity({"missing": name, "before": bindings["before"]}),
            snapshot_id=cid,
            disposition=DoctorRepairDisposition.SUPPORTED,
            observed_fact_refs=(
                content_identity(
                    {
                        "unbound_import": name,
                        "target": bindings["before"],
                        "caller": bindings["caller"],
                    }
                ),
            ),
            expected_behavior_refs=(
                content_identity(
                    {
                        "caller_requires_export": name,
                        "caller": bindings["caller"],
                        "fixture": bindings["fixture"],
                    }
                ),
            ),
            affected_symbol_refs=(name,),
            consumer_refs=("consumer:verified_seed_predecessor",),
            invalidation_refs=(cid,),
        )
        plan = DeterministicDoctorTactician().plan_finding(
            finding,
            current_roots=roots,
            candidates=(
                {
                    "candidate_ref": content_identity({"name": name, "donor": bindings["donor"]}),
                    "primary_signal": "exact_symbol",
                    "symbol_id": name,
                    "kind": "historical_immutable_declaration",
                    "semantic_authority": False,
                },
            ),
        )
        plans.append(plan.to_dict())
    save(output / "tactician.json", plans)
    # Plan admission is a mandatory gate, never inferred from a high vector score.
    if any(p["disposition"] != "planned" for p in plans):
        raise RuntimeError("tactician did not admit both plans; no mutation")
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    persisted = world.persist(
        {
            "task_id": "retained-seed-exports",
            "operation": "repair_candidate",
            "board": "qualification",
            "bindings": bindings,
            "graph_id": graph.graph_id,
            "index_id": vector_index.index_id,
            "proposal_only": True,
            "completion_authority": False,
        }
    )
    retrieved = world.records_for_decision(task_id="retained-seed-exports")
    save(output / "world-retrieval.json", retrieved)
    if retrieved["n"] != 1:
        raise RuntimeError("world memory hydration failed")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, path, ref in [
        ("ast", output / "ast.json", cid),
        ("knowledge_graph", output / "graph.json", graph.graph_id),
        ("vector", db, vector_index.index_id),
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
            subject_ref=TARGET,
            catalog_id=catalog["catalog_id"],
            record_kind=kind,
            record_ref=ref,
            project=False,
        )
    projection = meta.project_ducklake()
    save(output / "metadata-projection.json", projection)
    save(
        output / "metadata-retrieval.json",
        meta.compose_for_subject(subject_kind="path", subject_ref=TARGET),
    )
    if projection["status"] != "projected":
        raise RuntimeError("DuckLake metadata projection unavailable")
    if (root / CALLER).read_text() != caller or (root / FIXTURE).read_bytes() != fixture_source:
        raise RuntimeError("expectation source changed")
    lean = """def restore (missing : Bool) (old replacement : Nat) : Nat :=
  if missing then replacement else old
theorem preserves_existing (old replacement : Nat) : restore false old replacement = old := by rfl
theorem restores_missing (old replacement : Nat) : restore true old replacement = replacement := by rfl
#print axioms preserves_existing
#print axioms restores_missing
"""
    smt = """(set-logic QF_LIA)
(declare-const missing Bool)
(declare-const old Int)
(declare-const replacement Int)
(define-fun restored () Int (ite missing replacement old))
(assert (or (and (not missing) (distinct restored old)) (and missing (distinct restored replacement))))
(check-sat)
"""
    validation = """import importlib.util,sys,pytest
name='ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'
spec=importlib.util.spec_from_file_location(name,sys.argv[1]); module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
raise SystemExit(pytest.main(['-o','addopts=','--noconftest','-q','-o','log_cli=false','--tb=short','test/api/test_doctor_retained_seed_exports.py']))
"""
    result = apply_candidate(
        root,
        output,
        target=TARGET,
        before=before,
        after=after,
        index={
            "snapshot_id": cid,
            "scope": "current caller and explicit historical declaration fragments",
        },
        lean_source=lean,
        smt=smt,
        labels=(
            "theorem:literal-restoration",
            "property:preserve-bindings",
            "claim:missing-exports",
            "consequence:immutable-export-extension",
        ),
        assumptions=("assumption:AST-absence-and-literal-equality-checked-by-operator",),
        expected_axioms=[
            "'preserves_existing' does not depend on any axioms",
            "'restores_missing' does not depend on any axioms",
        ],
        proof_scope="abstract missing-binding selection only; Python AST checks and tests are separate evidence, not a whole-program proof",
        validation_code=validation,
    )
    result.update(
        operator_generated_candidate=True,
        tactician_plans=len(plans),
        vector_kind="lexical TF-IDF",
        ducklake_projected=True,
    )
    save(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="d65145f56")
    args = parser.parse_args()
    print(run(args.root.resolve(), args.output.resolve(), args.revision))
