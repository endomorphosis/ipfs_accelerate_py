"""Disposable actual Quack owner process. Never opens an existing board."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
from ipfs_accelerate_py.agent_supervisor.task_sources.spar_closeout_profile import SparCloseoutProfile
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection, STATUS_BOOTSTRAP_CLIENT_ID, TypedStateOwnerError)
from ipfs_accelerate_py.agent_supervisor.semantic_state import spar_native_verification as native


def run(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(Path("/dev/shm")):
        raise ValueError("qualification requires its explicitly disposable tmpfs root")
    material, facts, snapshot = json.loads((directory / "population.json").read_text())
    root = directory / "source"
    server = build_server(database_path=directory / "test.duckdb", state_dir=directory / "owner",
        store_id="spar-native-verifier-fixture", secret_handle="handle:spar-native-verifier-fixture",
        allow_legacy_board_unstall=False)
    results, client = {}, None
    try:
        identity = server.start()
        results["owner_identity"] = dict(server._command_gateway.identity)
        results["owner_ready"] = server.ready()["ready"]
        connection = server._connection
        for goal in facts["relations"]["goals"]["rows"]:
            connection.execute("INSERT INTO goals (goal_cid,goal_alias,objective_id,parent_goal_cid,ordinal,title,status,created_at,updated_at,revision,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [goal["goal_cid"],goal["goal_alias"],"objective:spar",goal["parent_goal_cid"],1,
                 goal["title"],goal["status"],"now","now",goal["revision"],goal["body_json"]])
        for task in facts["relations"]["tasks"]["rows"]:
            connection.execute("INSERT INTO tasks (task_cid,task_alias,goal_cid,plan_cid,objective_id,ordinal,status,revision,priority,created_at,updated_at,identity_json,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [task["task_cid"],task["task_alias"],task["goal_cid"],material["plan_root_cid"],"objective:spar",1,
                 task["status"],task["revision"],"P0","now","now",json.dumps({"repository_tree_id":material["repository_tree_id"]}),
                 json.dumps({**json.loads(task["body_json"]),"board_namespace":material["board_namespace"]})])
        for receipt in snapshot["completion_projection"]["completion_receipts"]:
            connection.execute("INSERT INTO completion_receipts (receipt_cid,task_cid,goal_cid,completed_at,evidence_digest,body_json) VALUES (?,?,?,?,?,?)",
                [receipt["receipt_cid"],receipt["task_cid"],receipt["goal_cid"],"now",receipt["evidence_digest"],json.dumps(receipt["body"])])
        profile = SparCloseoutProfile(material, repository_root=str(root))
        cids = sorted(t["task_cid"] for t in material["tasks"])
        server.bind_database_status_scope(**{k:material[k] for k in ("board_namespace","plan_root_cid","repository_tree_id")},
                                         task_cids=cids, closeout_profile=profile)
        client = TypedStateOwnerConnection(socket_path=server.typed_command_socket_path(),
            token=server.typed_command_token_path().read_text().strip(), client_id=STATUS_BOOTSTRAP_CLIENT_ID,
            process_birth_id="birth:fixture-reader",store_id=identity.store_id,status_bootstrap=True)
        def view():
            return client.completion_closeout_snapshot(cids)["closeout_facts"]["completion_profile"]
        results["before"] = view()["native_source_verification"]
        results["kit"] = server.publish_spar_source_forest()
        results["verified"] = server.verify_spar_semantic_source()
        results["after"] = view()
        results["replay"] = server.verify_spar_semantic_source()
        producer = profile._native_source_verification
        store = producer.store
        current = store.current_state_root(producer.namespace)
        record = store.get(current["root_cid"])
        request = store.get(record["request_cid"])
        results["execution"] = record["execution"]
        results["request"] = request
        before_reads = connection.execute("SELECT count(*) FROM kit_coordination_blocks").fetchone()[0]
        view(); view()
        results["read_only"] = before_reads == connection.execute("SELECT count(*) FROM kit_coordination_blocks").fetchone()[0]
        try:
            client._request("verify_spar_semantic_source")
        except TypedStateOwnerError:
            results["rpc_writer_refused"] = True
        connection.execute("UPDATE goals SET revision=revision+1 WHERE goal_alias='SPAR-G000'")
        results["goal_drift_refused"] = not view()["native_source_verification"]["admitted"]
        connection.execute("UPDATE goals SET revision=revision-1 WHERE goal_alias='SPAR-G000'")
        connection.execute("UPDATE tasks SET revision=revision+1 WHERE task_alias='SPAR-000'")
        results["task_drift_refused"] = not view()["native_source_verification"]["admitted"]
        connection.execute("UPDATE tasks SET revision=revision-1 WHERE task_alias='SPAR-000'")
        program = root / "program.py"; original = program.read_bytes(); program.write_bytes(original+b"# drift\n")
        results["source_drift_refused"] = not view()["native_source_verification"]["admitted"]
        program.write_bytes(original)
        forged = copy.deepcopy(record); forged["execution"]["process_identity"]["pid"] += 1
        forged_cid = store.put(forged,codec="dag-json",replicate=False)["cid"]
        store.compare_and_swap_state_root(producer.namespace,expected_revision=current["revision"],expected_root_cid=current["root_cid"],
            new_root_cid=forged_cid,operation_id="fixture-forgery")
        results["forged_execution_refused"] = not view()["native_source_verification"]["admitted"]
        moved = store.current_state_root(producer.namespace)
        store.compare_and_swap_state_root(producer.namespace,expected_revision=moved["revision"],expected_root_cid=moved["root_cid"],
            new_root_cid=current["root_cid"],operation_id="fixture-restore-exact-record")
        # A newly instantiated observer with the same connection/JSON identity
        # cannot manufacture this living producer's completed child custody.
        new_observer = native.NativeSparVerification(producer.persistence, str(root), connection=connection,
            transaction_lock=server._owner_transaction_lock, owner_identity=server._command_gateway.identity)
        source,binding = producer._capture(connection,cids)
        results["json_issuer_replay_refused"] = not new_observer.observe(source,binding)["admitted"]
        results["unknown_goals_preserved"] = connection.execute("SELECT count(*) FROM goals WHERE status='active'").fetchone()[0] == 32
        # Force a new request by changing a native revision, then introduce a
        # real file edit only after the actual independent verifier exits.
        connection.execute("UPDATE goals SET revision=revision+1 WHERE goal_alias='SPAR-G000'")
        actual = native.execute_verifier
        def drift_after_child(request, archive):
            value = actual(request,archive)
            program.write_bytes(original+b"# drift after actual child\n")
            return value
        native.execute_verifier = drift_after_child
        try:
            server.verify_spar_semantic_source()
        except native.NativeVerificationUnavailable:
            results["execution_boundary_drift_refused"] = True
        finally:
            native.execute_verifier = actual; program.write_bytes(original)
        connection.execute("UPDATE goals SET revision=revision-1 WHERE goal_alias='SPAR-G000'")
        connection.execute("UPDATE state_servers SET status='stopped' WHERE server_id=?",[identity.server_id])
        results["stale_owner_refused"] = not producer.observe(source,binding)["admitted"]
    finally:
        if client is not None: client.close()
        server.stop()
    # Reopen through the actual native lifecycle, not a fabricated identity
    # row. Retained content cannot re-create a new owner's child execution.
    old_generation = identity.generation
    try:
        replacement = server.start()
        replacement_profile = SparCloseoutProfile(material, repository_root=str(root))
        server.bind_database_status_scope(**{k:material[k] for k in ("board_namespace","plan_root_cid","repository_tree_id")},
                                         task_cids=cids, closeout_profile=replacement_profile)
        assert server.publish_spar_source_forest()["admitted"]
        replacement_observer = native.NativeSparVerification(replacement_profile._kit_source_forest,str(root),
            connection=server._connection,transaction_lock=server._owner_transaction_lock,
            owner_identity=server._command_gateway.identity)
        source,binding = replacement_observer._capture(server._connection,cids)
        results["actual_owner_rollover_refused"] = (replacement.generation > old_generation
            and not replacement_observer.observe(source,binding)["admitted"])
    finally:
        server.stop()
    return results


if __name__ == "__main__":
    output = run(sys.argv[1])
    Path(sys.argv[1],"qualification.json").write_text(json.dumps(output,sort_keys=True))
