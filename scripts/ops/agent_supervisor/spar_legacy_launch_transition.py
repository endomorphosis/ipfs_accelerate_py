"""One explicit legacy-capture to current dependency-seal scope transition.

The native materializer must admit its launch amendment before this module is
called. Git and configuration checks qualify source continuity, not task or
callback acceptance. Original origins, cursor histories and bootstrap receipts
are retained. No worker protocol exposes this local owner operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

from . import spar_merge_owner as role

SCHEMA = "spar/native-legacy-launch-scope-transition@1"
MAX_SOURCE_BYTES = 16 * 1024 * 1024


def _require(value, reason):
    if not value:
        raise role.SparMergeOwnerError(reason)


def _cid_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _native_cid(value):
    return _cid_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode())


def _git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=30)
    _require(result.returncode == 0 and len(result.stdout) <= MAX_SOURCE_BYTES,
             "legacy launch source Git evidence unavailable")
    return result.stdout


def _blob(root, head, relative, *, current=False):
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts and str(path) == relative,
             "legacy launch source path is not canonical")
    mode = _git(root, "ls-tree", head, "--", relative).split(b" ", 1)[0]
    _require(mode in (b"100644", b"100755"), "legacy launch source is not a regular Git blob")
    body = _git(root, "show", head + ":" + relative)
    if current:
        from .spar_legacy_capture import _bytes
        _require(_bytes(root / path, MAX_SOURCE_BYTES) == body, "legacy launch source differs from committed bytes")
    return body


def _sealed(root, head, config, *, current=False):
    seal = role._decode(_blob(root, head, config["dependency_seal_path"], current=current))
    body = {key: value for key, value in seal.items() if key != "seal_cid"}
    _require(seal.get("seal_cid") == _native_cid(body) == config["dependency_seal_cid"],
             "legacy launch dependency seal does not rehash")
    if current:
        for field in ("control_file_sha256", "bootstrap_runtime_file_sha256"):
            hashes = seal.get(field)
            _require(type(hashes) is dict and 1 <= len(hashes) <= 512,
                     "legacy launch dependency source inventory is incomplete")
            for path, digest in hashes.items():
                _require(hashlib.sha256(_blob(root, head, path, current=True)).hexdigest() == digest,
                         "legacy launch dependency source differs from current seal")
    return seal


@dataclass(frozen=True)
class LegacyLaunchTransition:
    receipt: dict
    scope_bindings: tuple
    _context: dict

    def require_current(self):
        fresh = qualify_transition(**self._context)
        _require(fresh.receipt == self.receipt and fresh.scope_bindings == self.scope_bindings,
                 "legacy launch transition changed after source qualification")


def qualify_transition(*, board, paths, amendment, origin, scopes):
    """Qualify only a seal-identity change against original captured Git bytes."""
    from ipfs_accelerate_py.agent_supervisor.task_sources.launch_source_amendment import LaunchSourceAmendment

    _require(type(amendment) is LaunchSourceAmendment, "typed native launch amendment required")
    root = Path(board.repo_root).absolute()
    manifest = role.validate_manifest(origin["manifest"])
    old_head, new_head = manifest["source_commit"], amendment.launch_source_head
    _require(_git(root, "rev-parse", "HEAD").decode().strip() == new_head
             and _git(root, "rev-parse", "HEAD^{tree}").decode().strip() == amendment.launch_repository_tree_id
             and _git(root, "rev-parse", old_head + "^{tree}").decode().strip() == manifest["source_tree"],
             "legacy launch source generation differs")
    _require(not _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip(),
             "legacy launch source is not clean")
    _git(root, "merge-base", "--is-ancestor", old_head, new_head)
    config_path = Path(board.config_path).absolute().relative_to(root).as_posix()
    old_raw = _blob(root, old_head, config_path)
    new_raw = _blob(root, new_head, config_path, current=True)
    old, new = role._decode(old_raw), role._decode(new_raw)
    _require(_git(root, "branch", "--show-current").decode().strip() == new["merge_target_branch"],
             "legacy launch source branch differs")
    old_cid, new_cid = _cid_bytes(old_raw), _cid_bytes(new_raw)
    _require(new_cid == amendment.launch_config_cid and old_cid != new_cid,
             "legacy launch configuration identity differs")
    _require(set(old) == set(new) and {key for key in old if old[key] != new[key]} == {"dependency_seal_cid"},
             "legacy launch transition permits only a dependency seal identity change")
    _require(new == board.payload, "legacy launch board payload differs from committed configuration")
    _sealed(root, old_head, old)
    new_seal = _sealed(root, new_head, new, current=True)
    _require(new_seal["seal_cid"] == amendment.dependency_seal_cid,
             "legacy launch amendment dependency seal differs")
    source = origin["capture"]["source"]
    _require(source["head"] == old_head and source["tree"] == manifest["source_tree"]
             and source["config_sha256"] == old_cid.removeprefix("sha256:"),
             "legacy launch capture source differs")
    from .spar_legacy_capture import _json
    bootstrap = _json(paths["bootstrap_receipt"])
    body = {key: value for key, value in bootstrap.items() if key != "bootstrap_receipt_id"}
    _require(_native_cid(body) == bootstrap.get("bootstrap_receipt_id") == amendment.bootstrap_receipt_id
             and bootstrap["source_head"] == amendment.bootstrap_source_head
             and bootstrap["repository_tree_id"] == amendment.bootstrap_repository_tree_id
             and bootstrap["plan_root_cid"] == amendment.bootstrap_plan_root_cid,
             "legacy launch immutable bootstrap differs")
    _git(root, "merge-base", "--is-ancestor", bootstrap["source_head"], old_head)
    for name in ("taskboard", "objectives", "plan", "validator"):
        relative = new[name + "_path"]
        raw = _blob(root, new_head, relative, current=True)
        expected = bootstrap["source_identities"][name]
        _require(_cid_bytes(raw) == expected == getattr(amendment, "immutable_" + name + "_cid")
                 and _blob(root, bootstrap["source_head"], relative) == raw
                 and _blob(root, old_head, relative) == raw,
                 "legacy launch immutable bootstrap input differs")
    _require(_cid_bytes(_blob(root, bootstrap["source_head"], config_path))
             == bootstrap["source_identities"]["config"] == amendment.bootstrap_config_cid,
             "legacy launch historical bootstrap configuration differs")
    previous = manifest["scope_bindings"]
    _require(len(previous) == len(scopes) == board.max_lanes,
             "legacy launch retained lane population differs")
    pairs = []
    for index, (before, after) in enumerate(zip(previous, scopes)):
        _require(before["config_cid"] == old_cid and after["config_cid"] == new_cid
                 and before["plan_cid"] == after["plan_cid"] == amendment.bootstrap_plan_root_cid
                 and before["board_namespace"] == after["board_namespace"] == board.board_namespace
                 and before["lane_id"] == after["lane_id"] == str(index)
                 and {**before, "config_cid": new_cid} == after,
                 "legacy launch retained recovery namespace differs")
        coordinates = {key: manifest[key] for key in ("store_id", "repository_id", "target_branch")}
        pairs.append({"previous_scope_cid": role.recovery_scope_cid(**coordinates, scope_binding=before),
                      "current_scope_cid": role.recovery_scope_cid(**coordinates, scope_binding=after)})
    # Only the distinct stopped-state protocol carries full native task facts.
    # Its local origin has a separate bound; legacy and recovery RPC identities
    # retain their existing admission and size limits.
    if origin.get("schema") == "spar/native-stopped-queue-origin@1":
        from .spar_stopped_capture import evidence_cid
        origin_cid = evidence_cid(origin)
    else:
        origin_cid = role._cid(origin)
    receipt = {"schema": SCHEMA, "manifest_cid": role._cid(manifest),
               "origin_cid": origin_cid, "bootstrap_receipt_id": amendment.bootstrap_receipt_id,
               "old_config_cid": old_cid, "current_config_cid": new_cid, "scope_pairs": pairs,
               "configuration_delta": ["dependency_seal_cid"],
               "callback_settled": False, "signing_authority": False, "completion_authority": False}
    context = dict(board=board, paths=dict(paths), amendment=amendment,
                   origin=role._decode(role._json(origin)), scopes=[dict(scope) for scope in scopes])
    return LegacyLaunchTransition(receipt, tuple(dict(scope) for scope in scopes), context)


def provision_transition(server, prepared):
    """Seed one successor configuration from exact retained initial cursors.

    The fixed migration identity allows restart replay, including later cursor
    heads, but refuses a second configuration transition. Extending this to a
    later configuration needs a separately reviewed current-head transition.
    """
    from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import _json as recovery_json

    transition = prepared.launch_transition
    _require(type(transition) is LegacyLaunchTransition, "qualified native launch transition required")
    transition.require_current()
    manifest = prepared.manifest
    _require(transition.receipt["manifest_cid"] == prepared.manifest_cid,
             "legacy launch transition changed its preserved manifest")
    expected_imports = {item["scope_cid"]: dict(item["cursors"]) for item in prepared.cursor_imports}
    imports = []
    with server._owner_transaction_lock:
        for pair in transition.receipt["scope_pairs"]:
            rows = server._connection.execute(
                "SELECT revision,state_cid,cursors_json FROM legacy_merge_recovery_cursors WHERE scope_cid=?",
                [pair["previous_scope_cid"]]).fetchall()
            expected = expected_imports.get(pair["previous_scope_cid"], {stage: "" for stage in role.STAGES})
            _require(len(rows) == 1 and tuple(rows[0][i] for i in range(3)) == (0, role._cid(expected), recovery_json(expected)),
                     "legacy launch original cursor advanced or differs from retained import")
            imports.append({"scope_cid": pair["current_scope_cid"], "cursors": expected,
                            "state_cid": role._cid(expected)})
        return server.provision_legacy_merge_recovery_schema(
            repository_id=manifest["repository_id"], target_branch=manifest["target_branch"],
            migration_id="spar-native-launch:" + prepared.manifest_cid,
            scope_bindings=list(transition.scope_bindings), cursor_imports=imports,
            receipt_imports=[{"receipt_key": "native-launch-scope-transition:" + prepared.manifest_cid,
                             "revision": 1, "receipt_cid": role._cid(transition.receipt),
                             "receipt": transition.receipt}])
