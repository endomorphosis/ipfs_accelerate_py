#!/usr/bin/env python3
"""Build the write-once PGIR-205 successor-v1 campaign input freeze.

This freeze supersedes the historical PGIR-014 root without mutating it.
Learned descendant execution stays closed unless every named gate passes.
A portability_no_go is recorded as a documented no-go, never as execution
authority.  Existing freeze bytes are never replaced.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FREEZE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FREEZE_DIR.parents[4]
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"
HISTORICAL_FREEZE = REPOSITORY_ROOT / "data/agent_supervisor/proof_grounded_ir_learning/freeze"
SUCCESSOR_BOARD = (
    REPOSITORY_ROOT / "docs/architecture/proof_grounded_ir_learning/successor.todo.md"
)
PGIR_211_VERIFIER = (
    REPOSITORY_ROOT
    / "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py"
)

HISTORICAL_CAMPAIGN_ROOT_SCHEMA = "IRCampaignInputRoot@1"
CAMPAIGN_ROOT_SCHEMA = "IRCampaignInputRoot@2"
REVISION_SET_SCHEMA = "PGIRDescendantTaskRevisionSet@1"
VERIFICATION_RECEIPT_SCHEMA = "PGIRFreezeVerificationReceipt@1"
MANIFEST_SCHEMA = "PGIRSuccessorFreezeBundleManifest@1"
RESULT_SCHEMA = "pgir-task-result@1"

HISTORICAL_ROOT_CID = "baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q"
HISTORICAL_ROOT_SHA256 = (
    "sha256:8a8cfcf2ebf7b735bcaf401299dd7806c3dff970c9bb81fdb199e0696a2cf575"
)
P208_CID = "baguqeeraburgmpdfo6weea57zlgkmppv7r34v2v3zstrepudxhrj6zrlgabq"
P209_CID = "baguqeerauh6r5lk47ecfmu5zjujmadrjiohd2ixkczcnurc33izkkvf2nb7q"
P210_CID = "baguqeera4ruaxwivpst2iwslorrgmbpuva6jqxczyjf6uditxg62atltnvkq"
P211_CID = "baguqeeram562re6snweb5nuinwprb4ehccvkin7kpylihktizlirsss7pllq"
P203_POLICY_CID = "baguqeeraluqwxtejicycax65cicqfibtyy5kxkxxyng5nzbxmuooqedw2vka"
P203_RESULT_CID = "baguqeera2wxw5woodrk5534uu6wzwhxu7e3c3glizd4ns6bofd32qx43yhbq"
P204_RETIREMENT_CID = "baguqeeraw4nh2c7xxamku4juzt5257krzlzuaxe64vl5cuz4h4c4iwm6xdjq"

OBJECTIVE_REVISION = "baguqeeraryux6dv5yim2by7j7zeffinnw5kfbl7prfiuew2yk6a2uknhnuiq"
REPOSITORY_ID = (
    "repository:sha256:3df67b4e7399635ecc20dc65888405eda8c32c7c28053e691fce8aa2aacaff4b"
)
POLICY_ID = "policy:implementation-daemon"
POLICY_REVISION = (
    "sha256:59a2d87bec0260f09bc675004e693abac2ba426c9e7662ed34790bc8f8f1b2ff"
)
HIDDEN_TEST_COMMITMENT = (
    "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
)
IR_CONFORMANCE_REQUIREMENT_ID = "287667496524558776121661391058779883318"

REQUIRED_HOLDOUTS = (
    "compiler",
    "cross_reference",
    "domain",
    "exception",
    "length",
    "lineage",
    "notation",
    "premise",
    "proof_library",
    "publication",
    "rare_operator",
    "time",
    "type",
)
INHERITED_BINDING_NAMES = (
    "compiler",
    "decompiler",
    "example_contracts",
    "gap_matrix",
    "schema_registry",
    "source_snapshots",
    "policy",
)
DESCENDANT_TASK_IDS = ("PGIR-206", "PGIR-207")

SAFE_PYTHONPATH = (
    "/home/barberb/.local/lib/python3.12/site-packages:"
    "/usr/local/lib/python3.12/dist-packages:"
    "/usr/lib/python3/dist-packages"
)
VERIFIER_ENVIRONMENT = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": SAFE_PYTHONPATH,
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "TZ": "UTC",
}

PGIR_211_IMPLEMENTATION = "59cdab5572cac092fb42398fe908a424e54d9c4e"
PGIR_211_MERGE = "d48759e84dadff1d0dec2e43ee8ad19c534682d7"
PGIR_211_COMPLETION = "20ef9e48d59b505b04e6236a9a31aaba287c36fd"
PGIR_211_COMPLETION_TREE = "dcc71e6806dc12d32b0ae5f7655a60572c40ec30"
PGIR_211_TARGET = "75791d58beeab140c2a3ebaf9789705b3e75c151"
NESTED_CURRENT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
NESTED_CURRENT_TREE = "7169c2a67929044a02350bc26d0a51c853a4981b"
SUCCESSOR_BOARD_BLOB = "fd110a44a8b530ba762bf2a702a77023514c2cb2"
TASK_IDENTITY_SOURCE = "ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py"
TASK_IDENTITY_SOURCE_BLOB = "5199029eae43a471591dadc62f59a94336ebc866"
PGIR_211_VERIFIER_BLOB = "712be25b94e24cfa2e53a02140bb1885210103c5"
CURRENT_TASK_IDENTITIES = {
    "PGIR-205": (
        "task/v1/8e297f0ebdc219a0e3e9fe4852a1adb75450afef8951425b585781aa29a76d11",
        "baguqeeraryux6dv5yim2by7j7zeffinnw5kfbl7prfiuew2yk6a2uknhnuiq",
    ),
    "PGIR-206": (
        "task/v1/2e49bb5ee8731dc89e9701a9456fa3e4d3c5c5fa4ce89f24486ddae50b3414c4",
        "baguqeerafze3wxxiomo4rhuxaguuk35d4tj4lrp2jtuj6jcinxnokczuctca",
    ),
    "PGIR-207": (
        "task/v1/efe4a7c323386342928737eaf8f18cad23f597f16cc27cd4acf32939066a3279",
        "baguqeera57skpqzdhbrufeuhg7vpr4mmvur7lf7rntbhzvfm6mutsbtkgj4q",
    ),
}
INHERITED_PORTABILITY_MISSING_OUTER = (
    "04fbb09b4a8b34e77d11bd8da6642e0978baa02c",
    "597a0285738c5878eed462593fd75e18715ff7f8",
)


class FreezeBuildError(RuntimeError):
    """Raised when the successor freeze cannot be issued without invention."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeBuildError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise FreezeBuildError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise FreezeBuildError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rendered_bytes(value: Any) -> bytes:
    validate_value(value)
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def dag_json_cid(value: Any) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).digest()
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + digest).decode(
        "ascii"
    ).rstrip("=").lower()


def raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode(
        "ascii"
    ).rstrip("=").lower()


def supervisor_identity(namespace: str, value: Any) -> str:
    return dag_json_cid({"namespace": namespace, "value": value})


def add_projection_identity(
    payload: Mapping[str, Any], *, cid_field: str, sha_field: str | None = None
) -> dict[str, Any]:
    result = dict(payload)
    projection = dict(result)
    if sha_field:
        result[sha_field] = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    result[cid_field] = dag_json_cid(projection)
    return result


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or absent JSON: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                FreezeBuildError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                FreezeBuildError(f"non-finite number {raw!r} in {path}")
            ),
        )
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    validate_value(value)
    return value


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    process = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=VERIFIER_ENVIRONMENT,
    )
    if process.returncode:
        raise FreezeBuildError(
            f"git {' '.join(args)} failed in {cwd}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def git_object(commit: str, path: str, *, cwd: Path) -> str:
    value = git("rev-parse", f"{commit}:{path}", cwd=cwd)
    require(bool(re.fullmatch(r"[0-9a-f]{40}", value)), f"invalid Git object {commit}:{path}")
    return value


def git_blob_oid(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def commit_record(commit: str, *, cwd: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    require(git("cat-file", "-t", commit, cwd=cwd) == "commit", f"not a commit: {commit}")
    tree = git("rev-parse", f"{commit}^{{tree}}", cwd=cwd)
    parents = git("show", "-s", "--format=%P", commit, cwd=cwd).split()
    subject = git("show", "-s", "--format=%s", commit, cwd=cwd)
    return {"commit": commit, "tree": tree, "parents": parents, "subject": subject}


def gitlink(commit: str) -> str:
    line = git("ls-tree", commit, "ipfs_datasets_py")
    parts = line.split()
    require(len(parts) >= 3 and parts[0] == "160000", f"missing gitlink at {commit}")
    return parts[2]


def file_binding(
    relative_path: str,
    *,
    repository: str,
    commit: str | None = None,
) -> dict[str, Any]:
    normalized = relative_path.replace("\\", "/").lstrip("./")
    require(
        bool(normalized) and not normalized.startswith("/") and ".." not in Path(normalized).parts,
        f"unsafe file binding path: {relative_path!r}",
    )
    path = REPOSITORY_ROOT / normalized
    require(path.is_file() and not path.is_symlink(), f"bound file is absent or unsafe: {normalized}")
    data = path.read_bytes()
    binding: dict[str, Any] = {
        "path": normalized,
        "repository": repository,
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    if commit:
        if repository == "ipfs_datasets_py":
            cwd = DATASETS_ROOT
            object_path = normalized.removeprefix("ipfs_datasets_py/")
        elif repository == "ipfs_accelerate_py":
            cwd = REPOSITORY_ROOT
            object_path = normalized
        else:
            raise FreezeBuildError(f"unsupported Git repository {repository!r}")
        binding["git_blob"] = git_object(commit, object_path, cwd=cwd)
        binding["revision"] = commit
        require(
            git_blob_oid(data) == binding["git_blob"],
            f"disk bytes differ from bound Git blob: {normalized}",
        )
    return binding


def binding_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return add_projection_identity(payload, cid_field="binding_cid")


def normalize_identity_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_identity_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text).rstrip("/")


def split_csv(value: str) -> list[str]:
    return [
        item
        for raw in str(value or "").split(",")
        if (item := raw.strip()) and item.lower() not in {"none", "n/a"}
    ]


def task_identity(task: Mapping[str, Any], *, semantic_key: str = "") -> dict[str, str]:
    """Use the exact current supervisor identity API on pinned source bytes."""

    source_path = REPOSITORY_ROOT / TASK_IDENTITY_SOURCE
    require(source_path.is_file() and not source_path.is_symlink(), "task identity API is absent")
    require(
        git_object(PGIR_211_COMPLETION, TASK_IDENTITY_SOURCE, cwd=REPOSITORY_ROOT)
        == TASK_IDENTITY_SOURCE_BLOB,
        "pinned task identity source blob drift",
    )
    require(
        git_blob_oid(source_path.read_bytes()) == TASK_IDENTITY_SOURCE_BLOB,
        "live task identity source differs from the PGIR-205 baseline",
    )
    module_name = "_pgir205_pinned_task_identity"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    require(spec is not None and spec.loader is not None, "cannot load task identity API")
    sys.dont_write_bytecode = True
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    metadata = dict(task.get("metadata") or {})
    if semantic_key:
        metadata["semantic key"] = semantic_key
    identity = module.canonical_task_identity(
        {
            "task_id": task.get("task_id"),
            "title": task.get("title"),
            "outputs": list(task.get("outputs") or ()),
            "acceptance": task.get("acceptance"),
            "metadata": metadata,
        },
        board_namespace=metadata.get("board namespace", "") or SUCCESSOR_BOARD.name,
        source_path=SUCCESSOR_BOARD,
    )
    return {
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "semantic_fingerprint": identity.semantic_fingerprint,
    }


def parse_successor_board() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    current_id = ""
    current_title = ""
    block: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, block
        if not current_id:
            return
        metadata: dict[str, str] = {}
        for line in block:
            stripped = line.strip()
            if stripped.startswith("- ") and ":" in stripped:
                key, value = stripped[2:].split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        task = {
            "task_id": current_id,
            "title": current_title,
            "depends_on": split_csv(metadata.get("depends on", "")),
            "outputs": split_csv(metadata.get("outputs", "")),
            "acceptance": metadata.get("acceptance", ""),
            "metadata": metadata,
        }
        task.update(task_identity(task))
        require(current_id not in tasks, f"duplicate task {current_id}")
        tasks[current_id] = task
        current_id = ""
        current_title = ""
        block = []

    require(
        git_object(
            PGIR_211_COMPLETION,
            "docs/architecture/proof_grounded_ir_learning/successor.todo.md",
            cwd=REPOSITORY_ROOT,
        )
        == SUCCESSOR_BOARD_BLOB,
        "pinned successor board blob drift",
    )
    board_text = git(
        "show",
        f"{PGIR_211_COMPLETION}:docs/architecture/proof_grounded_ir_learning/successor.todo.md",
    )
    for line in board_text.splitlines():
        if line.startswith("## "):
            flush()
            if line.startswith("## PGIR-"):
                header = line[3:].strip()
                current_id, _, current_title = header.partition(" ")
            continue
        if current_id:
            block.append(line)
    flush()
    require("PGIR-205" in tasks, "PGIR-205 is absent from the successor board")
    for task_id in DESCENDANT_TASK_IDS:
        require(task_id in tasks, f"{task_id} is absent from the successor board")
    return tasks


def transitive_descendants(
    tasks: Mapping[str, Mapping[str, Any]], ancestor: str
) -> tuple[str, ...]:
    children: dict[str, list[str]] = {}
    for task_id, task in tasks.items():
        for dependency in task["depends_on"]:
            children.setdefault(dependency, []).append(task_id)
    ordered: list[str] = []
    seen: set[str] = set()
    stack = list(children.get(ancestor, []))
    while stack:
        task_id = stack.pop(0)
        if task_id in seen:
            continue
        seen.add(task_id)
        ordered.append(task_id)
        stack.extend(children.get(task_id, []))
    return tuple(sorted(ordered, key=lambda item: (len(item), item)))


def disk_bytes(relative_path: str) -> bytes:
    path = REPOSITORY_ROOT / relative_path
    require(path.is_file() and not path.is_symlink(), f"required file absent: {relative_path}")
    return path.read_bytes()


def verify_historical_root(historical: Mapping[str, Any]) -> None:
    require(
        historical["schema"] == HISTORICAL_CAMPAIGN_ROOT_SCHEMA,
        "historical schema drift",
    )
    require(historical["task_id"] == "PGIR-014", "historical task drift")
    require(historical["root_cid"] == HISTORICAL_ROOT_CID, "historical root CID drift")
    require(historical["root_sha256"] == HISTORICAL_ROOT_SHA256, "historical root SHA drift")
    require(historical["supersession"]["previous_root_cid"] is None, "historical freeze is not the genesis root")
    require(historical["qualification"]["decision"] == "no_go", "historical freeze is not no-go")
    require(
        historical["qualification"]["descendant_execution_authorized"] is False,
        "historical freeze unexpectedly authorizes descendants",
    )
    projection = {key: value for key, value in historical.items() if key not in {"root_cid", "root_sha256"}}
    require(historical["root_cid"] == dag_json_cid(projection), "historical root CID does not replay")
    require(
        historical["root_sha256"]
        == "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest(),
        "historical root SHA does not replay",
    )


def current_repository() -> dict[str, str]:
    head = git("rev-parse", "HEAD")
    nested = git("rev-parse", "HEAD", cwd=DATASETS_ROOT)
    require(
        git("merge-base", "--is-ancestor", PGIR_211_COMPLETION, head) == "",
        "current HEAD does not descend from the PGIR-205 baseline",
    )
    require(
        git("rev-parse", f"{PGIR_211_COMPLETION}^{{tree}}")
        == PGIR_211_COMPLETION_TREE,
        "PGIR-205 baseline tree drift",
    )
    require(nested == NESTED_CURRENT, "nested checkout is not the integrated datasets commit")
    require(
        git("rev-parse", "HEAD^{tree}", cwd=DATASETS_ROOT) == NESTED_CURRENT_TREE,
        "nested checkout tree drift",
    )
    require(
        gitlink(PGIR_211_COMPLETION) == nested,
        "PGIR-205 baseline gitlink does not match nested HEAD",
    )
    return {
        "repository_id": REPOSITORY_ID,
        "source_revision": PGIR_211_COMPLETION,
        "source_tree_id": PGIR_211_COMPLETION_TREE,
        "datasets_commit": nested,
        "datasets_tree_id": NESTED_CURRENT_TREE,
        "source_set_id": "SRCSET-1",
    }


def pgir_211_forest() -> dict[str, Any]:
    implementation = commit_record(PGIR_211_IMPLEMENTATION)
    merge = commit_record(PGIR_211_MERGE)
    completion = commit_record(PGIR_211_COMPLETION)
    require(implementation["parents"] == ["10c38f56803442c224d379cedf6b0e5ca1e35147"], "PGIR-211 implementation parent drift")
    require(
        merge["parents"] == ["8cbb26404298a6f0b34e65c363444beb075e3dbe", PGIR_211_IMPLEMENTATION],
        "PGIR-211 merge parent drift",
    )
    require(completion["parents"] == [PGIR_211_MERGE], "PGIR-211 completion parent drift")
    require(completion["subject"] == "PGIR-211: mark todo completed", "PGIR-211 completion subject drift")
    require(git("merge-base", "--is-ancestor", PGIR_211_TARGET, PGIR_211_COMPLETION) == "", "PGIR-211 completion does not descend from RESULT(PGIR-211) target")
    require(git("merge-base", "--is-ancestor", PGIR_211_IMPLEMENTATION, PGIR_211_MERGE) == "", "implementation is not an ancestor of merge")
    require(git("merge-base", "--is-ancestor", PGIR_211_MERGE, PGIR_211_COMPLETION) == "", "merge is not an ancestor of completion")
    rows = []
    for role, record in (
        ("implementation", implementation),
        ("merge", merge),
        ("completion", completion),
    ):
        rows.append(
            {
                "task_id": "PGIR-211",
                "role": role,
                "commit": record["commit"],
                "tree": record["tree"],
                "parents": record["parents"],
                "subject": record["subject"],
                "gitlink": gitlink(record["commit"]),
            }
        )
        require(rows[-1]["gitlink"] == NESTED_CURRENT, f"PGIR-211 {role} gitlink drift")
    nested = commit_record(NESTED_CURRENT, cwd=DATASETS_ROOT)
    return {
        "schema": "proof-grounded-ir-learning/pgir-211-later-forest@1",
        "result_identity": "RESULT(PGIR-211)",
        "containing_commit_claimed": False,
        "circular_self_reference_avoided": True,
        "integrated_target": {
            "commit": PGIR_211_TARGET,
            "tree": git("rev-parse", f"{PGIR_211_TARGET}^{{tree}}"),
            "gitlink": gitlink(PGIR_211_TARGET),
        },
        "outer_commits": rows,
        "nested": {
            "commit": nested["commit"],
            "tree": nested["tree"],
            "parents": nested["parents"],
        },
        "note": "RESULT(PGIR-211) cannot name this later forest; PGIR-205 binds it at the freeze baseline.",
    }


def remote_contains(commit: str, *, cwd: Path) -> list[str]:
    process = subprocess.run(
        ("git", "for-each-ref", "--contains", commit, "--format=%(refname)", "refs/remotes"),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=VERIFIER_ENVIRONMENT,
    )
    if process.returncode:
        return []
    return [line for line in process.stdout.splitlines() if line.strip()]


def portability_probe(forest: Mapping[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    later = [row["commit"] for row in forest["outer_commits"]]
    inherited = list(INHERITED_PORTABILITY_MISSING_OUTER)
    missing_outer: list[str] = []
    for commit in (*inherited, *later):
        refs = remote_contains(commit, cwd=REPOSITORY_ROOT)
        if not refs:
            missing_outer.append(commit)
    missing_outer = sorted(set(missing_outer))
    nested_refs = remote_contains(NESTED_CURRENT, cwd=DATASETS_ROOT)
    missing_nested: list[str] = [] if nested_refs else [NESTED_CURRENT]
    observation_method = (
        "git for-each-ref --contains <commit> --format=%(refname) refs/remotes "
        "in the current outer and nested repositories; local stale remote-tracking "
        "refs are insufficient for a fresh recursive checkout"
    )
    status = "portability_no_go" if (missing_outer or missing_nested) else "portable"
    payload = {
        "schema": "proof-grounded-ir-learning/pgir-205-portability-receipt@1",
        "task_id": "PGIR-205",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "outer_commit": PGIR_211_COMPLETION,
            "outer_tree": PGIR_211_COMPLETION_TREE,
            "nested_gitlink": NESTED_CURRENT,
            "nested_tree": NESTED_CURRENT_TREE,
        },
        "status": status,
        "observation_method": observation_method,
        "inherited_pgir_211_missing_outer_commits": list(inherited),
        "later_forest_commits": later,
        "missing_outer_commits": missing_outer,
        "missing_nested_commits": missing_nested,
        "nested_current_remote_refs": nested_refs,
        "pgir_205_execution_authorized": False,
        "effect": (
            "PGIR-205 remains fail-closed: no materialization, training, promotion, "
            "or execution authority follows from this freeze."
        ),
    }
    return add_projection_identity(
        payload,
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )


def verify_projection_identity(
    value: Mapping[str, Any], *, cid_field: str, sha_field: str | None = None
) -> None:
    projection = dict(value)
    claimed_cid = projection.pop(cid_field, "")
    claimed_sha = projection.pop(sha_field, "") if sha_field else ""
    require(claimed_cid == dag_json_cid(projection), f"{cid_field} projection drift")
    if sha_field:
        expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
        require(claimed_sha == expected_sha, f"{sha_field} projection drift")


def load_sealed_capture(
    name: str, *, cid_field: str, sha_field: str | None = None
) -> dict[str, Any]:
    value = strict_json(FREEZE_DIR / name)
    verify_projection_identity(value, cid_field=cid_field, sha_field=sha_field)
    return value


def validate_portability_receipt(receipt: Mapping[str, Any]) -> None:
    verify_projection_identity(
        receipt,
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    require(receipt.get("task_id") == "PGIR-205", "portability task drift")
    require(receipt.get("status") == "portability_no_go", "portability verdict drift")
    require(receipt.get("pgir_205_execution_authorized") is False, "portability authorized execution")
    require(
        receipt.get("baseline")
        == {
            "outer_commit": PGIR_211_COMPLETION,
            "outer_tree": PGIR_211_COMPLETION_TREE,
            "nested_gitlink": NESTED_CURRENT,
            "nested_tree": NESTED_CURRENT_TREE,
        },
        "portability baseline drift",
    )
    require(
        set(receipt.get("missing_outer_commits") or ())
        >= set(INHERITED_PORTABILITY_MISSING_OUTER),
        "portability receipt omitted inherited missing commits",
    )
    require(receipt.get("missing_nested_commits") == [], "nested portability drift")


def run_pgir_211_verifier() -> dict[str, Any]:
    require(PGIR_211_VERIFIER.is_file() and not PGIR_211_VERIFIER.is_symlink(), "PGIR-211 verifier is absent")
    source = disk_bytes(
        "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py"
    )
    require(
        git_object(
            PGIR_211_COMPLETION,
            "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py",
            cwd=REPOSITORY_ROOT,
        )
        == PGIR_211_VERIFIER_BLOB,
        "pinned PGIR-211 verifier blob drift",
    )
    require(
        git_blob_oid(source) == PGIR_211_VERIFIER_BLOB,
        "live PGIR-211 verifier differs from the PGIR-205 baseline",
    )
    argv = [
        "/usr/bin/python3.12",
        "-S",
        "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py",
        "--network",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        env=VERIFIER_ENVIRONMENT,
        timeout=1200,
    )
    ended_at = datetime.now(timezone.utc).isoformat()
    stdout = process.stdout.decode("utf-8", errors="replace")
    stderr = process.stderr.decode("utf-8", errors="replace")
    require(process.returncode == 0, f"fresh PGIR-211 network verifier failed: {(stderr or stdout)[-2000:]}")
    require(stderr == "", "fresh PGIR-211 network verifier emitted stderr")
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(stdout.strip().splitlines()[-1] if stdout.strip() else "{}")
    except json.JSONDecodeError:
        parsed = None
    require(isinstance(parsed, dict), "PGIR-211 verifier did not emit a JSON object")
    require(parsed.get("verified") is True, "fresh PGIR-211 network verifier is not verified")
    require(parsed.get("component_verified") is True, "fresh PGIR-211 component replay failed")
    require(parsed.get("pgir_205_execution_authorized") is False, "fresh PGIR-211 run authorized PGIR-205")
    require(parsed.get("completion_authoritative") is False, "fresh PGIR-211 run claimed completion authority")
    require(parsed.get("decision") == "permanent_no_go", "fresh PGIR-211 run is not permanent_no_go")
    require(parsed.get("acceptance_cid") == P211_CID, "fresh PGIR-211 acceptance CID drift")
    live_network = parsed.get("live_network") or {}
    require(
        live_network.get("requested") == 21 and live_network.get("matched") == 21,
        "fresh PGIR-211 live network population is not 21/21",
    )
    require(
        live_network.get("receipt_replay_used_as_substitute") is False,
        "fresh PGIR-211 run substituted a frozen network receipt",
    )
    payload = {
        "schema": "proof-grounded-ir-learning/pgir-211-baseline-verifier-run@1",
        "task_id": "PGIR-205",
        "result_identity": "RESULT(PGIR-211)",
        "acceptance_cid": P211_CID,
        "mode": "network",
        "argv": argv,
        "cwd": ".",
        "exit_code": process.returncode,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "stdout": parsed,
        "stdout_raw_cid": raw_cid(process.stdout),
        "stdout_sha256": "sha256:" + hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stdout_size_bytes": len(process.stdout),
        "stderr_raw_cid": raw_cid(process.stderr),
        "stderr_sha256": "sha256:" + hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stderr_size_bytes": len(process.stderr),
        "stderr_empty": stderr == "",
        "pgir_205_execution_authorized": False,
        "baseline": {
            "outer_commit": PGIR_211_COMPLETION,
            "outer_tree": PGIR_211_COMPLETION_TREE,
            "nested_gitlink": NESTED_CURRENT,
            "nested_tree": NESTED_CURRENT_TREE,
        },
        "verifier_source": {
            "path": "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py",
            "revision": PGIR_211_COMPLETION,
            "git_blob": PGIR_211_VERIFIER_BLOB,
            "raw_cid": raw_cid(source),
            "sha256": "sha256:" + hashlib.sha256(source).hexdigest(),
            "size_bytes": len(source),
        },
        "environment": {
            "executable": "/usr/bin/python3.12",
            "no_site": True,
            "pythonpath": SAFE_PYTHONPATH,
            "path": "/usr/bin:/bin",
        },
        "note": "Fresh run at the PGIR-205 baseline; it does not circularly name the PGIR-211 completion commit as RESULT(PGIR-211).",
    }
    result = add_projection_identity(payload, cid_field="run_cid", sha_field="run_sha256")
    validate_pgir_211_verifier_run(result)
    return result


def validate_pgir_211_verifier_run(run: Mapping[str, Any]) -> None:
    verify_projection_identity(run, cid_field="run_cid", sha_field="run_sha256")
    expected_argv = [
        "/usr/bin/python3.12",
        "-S",
        "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py",
        "--network",
    ]
    expected_baseline = {
        "outer_commit": PGIR_211_COMPLETION,
        "outer_tree": PGIR_211_COMPLETION_TREE,
        "nested_gitlink": NESTED_CURRENT,
        "nested_tree": NESTED_CURRENT_TREE,
    }
    require(run.get("mode") == "network", "sealed PGIR-211 run is not network mode")
    require(run.get("argv") == expected_argv, "sealed PGIR-211 argv drift")
    require(run.get("cwd") == ".", "sealed PGIR-211 cwd drift")
    require(run.get("exit_code") == 0, "sealed PGIR-211 exit drift")
    require(run.get("stderr_empty") is True, "sealed PGIR-211 stderr was not empty")
    require(run.get("stderr_size_bytes") == 0, "sealed PGIR-211 stderr size drift")
    require(run.get("stderr_raw_cid") == raw_cid(b""), "sealed PGIR-211 stderr CID drift")
    require(
        run.get("stderr_sha256")
        == "sha256:" + hashlib.sha256(b"").hexdigest(),
        "sealed PGIR-211 stderr identity drift",
    )
    require(
        run.get("stdout_size_bytes") == len(canonical_bytes(run.get("stdout"))) + 1,
        "sealed PGIR-211 stdout size drift",
    )
    expected_stdout = canonical_bytes(run.get("stdout")) + b"\n"
    require(run.get("stdout_raw_cid") == raw_cid(expected_stdout), "sealed PGIR-211 stdout CID drift")
    require(
        run.get("stdout_sha256")
        == "sha256:" + hashlib.sha256(expected_stdout).hexdigest(),
        "sealed PGIR-211 stdout hash drift",
    )
    require(run.get("baseline") == expected_baseline, "sealed PGIR-211 baseline drift")
    started = datetime.fromisoformat(str(run.get("started_at_utc") or ""))
    ended = datetime.fromisoformat(str(run.get("ended_at_utc") or ""))
    require(started.tzinfo is not None and ended.tzinfo is not None and started <= ended, "sealed PGIR-211 observation interval drift")
    stdout = run.get("stdout") or {}
    require(stdout.get("verified") is True, "sealed PGIR-211 run is not verified")
    require(stdout.get("component_verified") is True, "sealed PGIR-211 component replay failed")
    require(stdout.get("acceptance_cid") == P211_CID, "sealed PGIR-211 acceptance CID drift")
    require(stdout.get("decision") == "permanent_no_go", "sealed PGIR-211 decision drift")
    require(stdout.get("completion_authoritative") is False, "sealed PGIR-211 run claimed completion")
    require(stdout.get("pgir_205_execution_authorized") is False, "sealed PGIR-211 run authorized PGIR-205")
    live_network = stdout.get("live_network") or {}
    require(live_network.get("requested") == 21 and live_network.get("matched") == 21, "sealed PGIR-211 live replay is not 21/21")
    require(live_network.get("receipt_replay_used_as_substitute") is False, "sealed PGIR-211 run used a receipt substitute")
    require(
        (run.get("verifier_source") or {}).get("git_blob") == PGIR_211_VERIFIER_BLOB,
        "sealed PGIR-211 verifier source blob drift",
    )


def evaluate_gates(
    *,
    rights: Mapping[str, Any],
    corpus: Mapping[str, Any],
    split: Mapping[str, Any],
    holdouts: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tokenizer_result: Mapping[str, Any],
    retirement: Mapping[str, Any],
    baseline: Mapping[str, Any],
    integrated: Mapping[str, Any],
    verifier_run: Mapping[str, Any],
    portability: Mapping[str, Any],
) -> dict[str, Any]:
    rights_ok = (
        rights.get("training_eligible") is True
        and int(rights.get("training_admitted_rows") or 0) > 0
        and rights.get("admission_decision") != "permanent_zero_for_jdao_pinset_1"
    )
    corpus_ok = (
        corpus.get("materialized") is True
        and int(corpus.get("counts", {}).get("admitted_source_rows") or 0) > 0
        and int(corpus.get("counts", {}).get("materialized_source_rows") or 0) > 0
    )
    holdout_map = holdouts.get("holdouts") or {}
    failed_holdouts = [
        name
        for name in REQUIRED_HOLDOUTS
        if (holdout_map.get(name) or {}).get("status") != "populated"
    ]
    holdouts_ok = not failed_holdouts and holdouts.get("all_declared_holdouts_resolved") is True
    tokenizer_ok = (
        tokenizer.get("status") == "admitted"
        and tokenizer.get("learned_tokenizer_admission", {}).get("admission_status") == "admitted"
        and tokenizer_result.get("training_authorized") is True
    )
    baseline_ok = (
        retirement.get("acceptance", {}).get("current_input_qualified_r1_cid") not in (None, "", False)
        and retirement.get("decision", {}).get("status") != "retired"
        and baseline.get("pgir_205_execution_authorized") is True
    )
    integrated_ok = (
        integrated.get("pgir_205_execution_authorized") is True
        and integrated.get("decision") == "admitted"
        and verifier_run.get("stdout", {}).get("pgir_205_execution_authorized") is True
    )
    portability_ok = portability.get("status") == "portable" and not portability.get(
        "missing_outer_commits"
    )
    gates = {
        "rights": {
            "passed": bool(rights_ok),
            "result_identity": "RESULT(PGIR-200)",
            "training_admitted_rows": rights.get("training_admitted_rows"),
            "training_eligible": rights.get("training_eligible"),
            "admission_decision": rights.get("admission_decision"),
        },
        "corpus": {
            "passed": bool(corpus_ok),
            "result_identity": "RESULT(PGIR-201)",
            "materialized": corpus.get("materialized"),
            "admitted_source_rows": corpus.get("counts", {}).get("admitted_source_rows"),
            "materialized_source_rows": corpus.get("counts", {}).get("materialized_source_rows"),
        },
        "holdouts": {
            "passed": bool(holdouts_ok),
            "result_identity": "RESULT(PGIR-202)",
            "required_holdouts": list(REQUIRED_HOLDOUTS),
            "failed_holdouts": failed_holdouts,
            "leakage_passed": split.get("leakage_passed") is True,
            "hidden_test_commitment": split.get("hidden_test_commitment"),
            "hidden_labels_accessed": False,
        },
        "tokenizer": {
            "passed": bool(tokenizer_ok),
            "result_identity": "RESULT(PGIR-203)",
            "result_cid": tokenizer_result.get("result_cid"),
            "status": tokenizer.get("status"),
            "policy_cid": tokenizer.get("policy_cid"),
            "training_authorized": tokenizer_result.get("training_authorized"),
        },
        "current_baseline": {
            "passed": bool(baseline_ok),
            "result_identity": "RESULT(PGIR-204)",
            "adjudication_identity": "RESULT(PGIR-209)",
            "retirement_cid": retirement.get("retirement_cid"),
            "decision": retirement.get("decision", {}).get("status"),
            "current_input_qualified_r1_cid": retirement.get("acceptance", {}).get(
                "current_input_qualified_r1_cid"
            ),
        },
        "integrated_evidence": {
            "passed": bool(integrated_ok),
            "result_identity": "RESULT(PGIR-211)",
            "acceptance_cid": integrated.get("acceptance_cid"),
            "decision": integrated.get("decision"),
            "completion_authoritative": integrated.get("completion_authoritative"),
            "pgir_205_execution_authorized": integrated.get("pgir_205_execution_authorized"),
            "fresh_verifier_run_cid": verifier_run.get("run_cid"),
            "fresh_verifier_mode": verifier_run.get("mode"),
        },
        "portability": {
            "passed": bool(portability_ok),
            "receipt_cid": portability.get("receipt_cid"),
            "status": portability.get("status"),
            "missing_outer_commits": portability.get("missing_outer_commits"),
            "missing_nested_commits": portability.get("missing_nested_commits"),
            "documented_no_go_required": portability.get("status") == "portability_no_go",
        },
    }
    require(not rights_ok, "rights gate unexpectedly passed")
    require(not corpus_ok, "corpus gate unexpectedly passed")
    require(not holdouts_ok, "holdout gate unexpectedly passed")
    require(not tokenizer_ok, "tokenizer gate unexpectedly passed")
    require(not baseline_ok, "current-baseline gate unexpectedly passed")
    require(not integrated_ok, "integrated-evidence gate unexpectedly passed")
    require(not portability_ok, "portability gate unexpectedly passed; do not invent a go")
    return gates


def reason_codes_from_gates(gates: Mapping[str, Any], portability: Mapping[str, Any]) -> list[str]:
    codes = [
        "no_rights_admitted_training_rows",
        "corpus_not_materialized",
        "required_holdouts_insufficient",
        "tokenizer_not_admitted",
        "historical_semantic_baseline_not_currently_qualified",
        "integrated_evidence_does_not_authorize_execution",
    ]
    if portability.get("status") == "portability_no_go":
        codes.append("portability_no_go")
    return codes


def build_campaign_root(
    *,
    historical: Mapping[str, Any],
    rights: Mapping[str, Any],
    corpus: Mapping[str, Any],
    lineage: Mapping[str, Any],
    split: Mapping[str, Any],
    holdouts: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tokenizer_result: Mapping[str, Any],
    retirement: Mapping[str, Any],
    baseline: Mapping[str, Any],
    integrated: Mapping[str, Any],
    forest: Mapping[str, Any],
    verifier_run: Mapping[str, Any],
    portability: Mapping[str, Any],
    gates: Mapping[str, Any],
    repository: Mapping[str, str],
) -> dict[str, Any]:
    nested = repository["datasets_commit"]
    outer = repository["source_revision"]
    successor_files = {
        "rights": file_binding(
            "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/rights_manifest.json",
            repository="ipfs_datasets_py",
            commit=nested,
        ),
        "corpus": file_binding(
            "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json",
            repository="ipfs_datasets_py",
            commit=nested,
        ),
        "lineage": file_binding(
            "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/lineage_graph.json",
            repository="ipfs_datasets_py",
            commit=nested,
        ),
        "split": file_binding(
            "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json",
            repository="ipfs_datasets_py",
            commit=nested,
        ),
        "holdouts": file_binding(
            "ipfs_datasets_py/data/ir_learning/splits/successor-v1/holdout_report.json",
            repository="ipfs_datasets_py",
            commit=nested,
        ),
        "tokenizer": file_binding(
            "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/tokenizer_policy.json",
            repository="ipfs_accelerate_py",
            commit=outer,
        ),
        "tokenizer_result": file_binding(
            "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/result.json",
            repository="ipfs_accelerate_py",
            commit=outer,
        ),
        "retirement": file_binding(
            "ipfs_datasets_py/data/ir_learning/evaluations/deterministic/successor-v1/retirement_receipt.json",
            repository="ipfs_datasets_py",
            commit=nested,
        ),
        "baseline": file_binding(
            "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/baseline-acceptance/baseline_acceptance.json",
            repository="ipfs_accelerate_py",
            commit=outer,
        ),
        "integrated": file_binding(
            "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/integrated_acceptance.json",
            repository="ipfs_accelerate_py",
            commit=outer,
        ),
        "historical_root": file_binding(
            "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json",
            repository="ipfs_accelerate_py",
            commit=outer,
        ),
    }
    require(tokenizer["policy_cid"] == P203_POLICY_CID, "PGIR-203 policy CID drift")
    require(tokenizer_result["result_cid"] == P203_RESULT_CID, "PGIR-203 result CID drift")
    require(retirement["retirement_cid"] == P204_RETIREMENT_CID, "PGIR-204 retirement CID drift")
    require(baseline["acceptance_cid"] == P209_CID, "PGIR-209 acceptance CID drift")
    require(integrated["acceptance_cid"] == P211_CID, "PGIR-211 acceptance CID drift")
    require(split["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT, "hidden-test commitment drift")
    require(holdouts["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT, "holdout hidden-test commitment drift")
    require(int(rights["training_admitted_rows"]) == 0, "rights admitted rows drifted")
    require(int(rights["quarantined_source_record_count"]) == 7173, "quarantine count drifted")
    require(corpus["counts"]["admitted_source_rows"] == 0, "corpus admitted rows drifted")
    require(corpus["materialized"] is False, "corpus unexpectedly materialized")
    require(lineage["validation"]["materialized_row_count"] == 0, "lineage materialized rows drifted")

    bindings: dict[str, Any] = {}
    for name in INHERITED_BINDING_NAMES:
        inherited = dict(historical["bindings"][name])
        require("binding_cid" in inherited, f"historical {name} binding CID missing")
        bindings[name] = inherited
    bindings["rights"] = binding_identity(
        {
            "task_id": "PGIR-200",
            "result_identity": "RESULT(PGIR-200)",
            "admission_decision": rights["admission_decision"],
            "training_admitted_rows": rights["training_admitted_rows"],
            "quarantined_source_count": rights["quarantined_source_record_count"],
            "pinset_id": rights["pinset_id"],
            "file": successor_files["rights"],
        }
    )
    bindings["corpus"] = binding_identity(
        {
            "task_id": "PGIR-201",
            "result_identity": "RESULT(PGIR-201)",
            "manifest_id": corpus["manifest_id"],
            "materialized": corpus["materialized"],
            "source_count": 7173,
            "training_admitted_rows": corpus["counts"]["admitted_source_rows"],
            "materialized_source_rows": corpus["counts"]["materialized_source_rows"],
            "file": successor_files["corpus"],
        }
    )
    bindings["lineage"] = binding_identity(
        {
            "task_id": "PGIR-201",
            "result_identity": "RESULT(PGIR-201)",
            "graph_id": lineage["graph_id"],
            "admitted_lineage_groups": lineage["admitted_lineage_groups"],
            "materialized_row_count": lineage["validation"]["materialized_row_count"],
            "file": successor_files["lineage"],
        }
    )
    bindings["split"] = binding_identity(
        {
            "task_id": "PGIR-202",
            "result_identity": "RESULT(PGIR-202)",
            "status": split["status"],
            "leakage_passed": split["leakage_passed"],
            "hidden_test_commitment": split["hidden_test_commitment"],
            "holdouts": {
                name: {
                    "status": holdout_map["status"],
                    "count": holdout_map["count"],
                }
                for name, holdout_map in (holdouts["holdouts"].items())
                if name in REQUIRED_HOLDOUTS
            },
            "file": successor_files["split"],
            "holdout_report": successor_files["holdouts"],
        }
    )
    bindings["tokenizer_policy"] = binding_identity(
        {
            "task_id": "PGIR-203",
            "result_identity": "RESULT(PGIR-203)",
            "policy_cid": tokenizer["policy_cid"],
            "result_cid": tokenizer_result["result_cid"],
            "status": tokenizer["status"],
            "unknown_token_behavior": tokenizer["unknown_token_behavior"],
            "training_authorized": tokenizer_result["training_authorized"],
            "file": successor_files["tokenizer"],
            "result_file": successor_files["tokenizer_result"],
        }
    )
    gates = {
        **gates,
        "current_baseline": {
            **gates["current_baseline"],
            "retirement_file": successor_files["retirement"],
            "baseline_file": successor_files["baseline"],
        },
        "integrated_evidence": {
            **gates["integrated_evidence"],
            "file": successor_files["integrated"],
        },
        "rights": {**gates["rights"], "file": successor_files["rights"]},
        "corpus": {**gates["corpus"], "file": successor_files["corpus"]},
        "holdouts": {
            **gates["holdouts"],
            "file": successor_files["split"],
            "holdout_report": successor_files["holdouts"],
        },
        "tokenizer": {
            **gates["tokenizer"],
            "file": successor_files["tokenizer"],
            "result_file": successor_files["tokenizer_result"],
        },
    }
    reasons = reason_codes_from_gates(gates, portability)
    insufficient = list(REQUIRED_HOLDOUTS)
    payload = {
        "schema": CAMPAIGN_ROOT_SCHEMA,
        "interface": "proof-grounded-ir-learning/campaign-input-root/v2",
        "contract_version": 2,
        "task_id": "PGIR-205",
        "objective_id": "PGIR-205",
        "objective_revision": OBJECTIVE_REVISION,
        "repository": dict(repository),
        "bindings": bindings,
        "referential_integrity": {
            "all_required_bindings_present": True,
            "required_binding_names": sorted(bindings),
            "unresolved_identities": [],
            "compiler_alias_resolved": True,
            "decompiler_alias_resolved": True,
            "hidden_labels_accessed": False,
            "source_or_split_mutated": False,
        },
        "gates": gates,
        "integrated_evidence": {
            "result_identity": "RESULT(PGIR-211)",
            "acceptance_cid": P211_CID,
            "predecessor_acceptance_cids": {
                "PGIR-208": P208_CID,
                "PGIR-209": P209_CID,
                "PGIR-210": P210_CID,
            },
            "supersession_chain": ["PGIR-208", "PGIR-210", "PGIR-211"],
            "decision": integrated["decision"],
            "completion_authoritative": integrated["completion_authoritative"],
            "pgir_205_execution_authorized": False,
            "file": successor_files["integrated"],
            "later_forest": forest,
            "fresh_verifier_run": {
                "run_cid": verifier_run["run_cid"],
                "mode": verifier_run["mode"],
                "exit_code": verifier_run["exit_code"],
                "stdout": verifier_run["stdout"],
            },
        },
        "qualification": {
            "decision": "no_go",
            "lease_barrier": "closed",
            "descendant_execution_authorized": False,
            "training_task_eligible_count": 0,
            "training_admitted_rows": 0,
            "rights_quarantined_rows": 7173,
            "corpus_materialized": False,
            "leakage_passed": True,
            "insufficient_holdouts": insufficient,
            "reason_codes": reasons,
        },
        "canonicalization": {
            "identity_projection": "entire document excluding root_cid and root_sha256",
            "json": "UTF-8; sorted keys; compact separators; ensure_ascii=false; no floats",
            "cid": "CIDv1/base32/dag-json/sha2-256",
            "rendering": "two-space indentation and one terminal LF",
        },
        "supersession": {
            "immutable": True,
            "previous_root_cid": HISTORICAL_ROOT_CID,
            "previous_root_schema": HISTORICAL_CAMPAIGN_ROOT_SCHEMA,
            "previous_root_path": "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json",
            "previous_root_file": successor_files["historical_root"],
            "replacement_policy": "supersede_never_overwrite",
        },
    }
    require(payload["supersession"]["previous_root_cid"] == historical["root_cid"], "previous_root_cid does not bind the historical freeze")
    require(all(not gate["passed"] for gate in gates.values()), "a gate passed; eligibility would require all gates")
    return add_projection_identity(payload, cid_field="root_cid", sha_field="root_sha256")


def build_descendant_revisions(root: Mapping[str, Any]) -> dict[str, Any]:
    tasks = parse_successor_board()
    descendant_ids = transitive_descendants(tasks, "PGIR-205")
    require(tuple(descendant_ids) == DESCENDANT_TASK_IDS, "PGIR-205 descendant population drifted")
    semantic_key = f"pgir-campaign-input-root@2:{root['root_cid']}"
    revisions: list[dict[str, Any]] = []
    for task_id in descendant_ids:
        task = tasks[task_id]
        expected_key, expected_cid = CURRENT_TASK_IDENTITIES[task_id]
        require(task["canonical_task_key"] == expected_key, f"{task_id} current task key drift")
        require(task["canonical_task_cid"] == expected_cid, f"{task_id} current task CID drift")
        revised = task_identity(task, semantic_key=semantic_key)
        revisions.append(
            {
                "task_id": task_id,
                "title": task["title"],
                "depends_on": task["depends_on"],
                "current_task_cid": task["canonical_task_cid"],
                "current_task_key": task["canonical_task_key"],
                "revised_task_cid": revised["canonical_task_cid"],
                "revised_task_key": revised["canonical_task_key"],
                "semantic_fingerprint": revised["semantic_fingerprint"],
                "input_binding": {
                    "campaign_input_root_cid": root["root_cid"],
                    "semantic_key": semantic_key,
                    "decision": root["qualification"]["decision"],
                },
                "lease_eligible": False,
                "block_reason_codes": root["qualification"]["reason_codes"],
            }
        )
    graph_projection = {
        "schema": "PGIRDescendantTaskGraph@1",
        "root_task_id": "PGIR-205",
        "tasks": [
            {
                "task_id": item["task_id"],
                "task_cid": item["revised_task_cid"],
                "depends_on": item["depends_on"],
            }
            for item in revisions
        ],
    }
    payload = {
        "schema": REVISION_SET_SCHEMA,
        "task_id": "PGIR-205",
        "campaign_input_root_cid": root["root_cid"],
        "source_plan_task_cid": tasks["PGIR-205"]["canonical_task_cid"],
        "source_plan_task_key": tasks["PGIR-205"]["canonical_task_key"],
        "identity_source": {
            "board_revision": PGIR_211_COMPLETION,
            "board_blob": SUCCESSOR_BOARD_BLOB,
            "task_identity_revision": PGIR_211_COMPLETION,
            "task_identity_blob": TASK_IDENTITY_SOURCE_BLOB,
        },
        "revision_patch": {
            "metadata_field": "Semantic key",
            "metadata_value": semantic_key,
            "protected_board_mutated": False,
            "application_policy": "supervisor_compare_and_swap_after_superseding_admission",
        },
        "candidate_graph_cid": dag_json_cid(graph_projection),
        "descendant_task_count": len(revisions),
        "lease_eligible_count": 0,
        "revisions": revisions,
    }
    expected_plan_key, expected_plan_cid = CURRENT_TASK_IDENTITIES["PGIR-205"]
    require(payload["source_plan_task_key"] == expected_plan_key, "PGIR-205 task key drift")
    require(payload["source_plan_task_cid"] == expected_plan_cid, "PGIR-205 task CID drift")
    require(payload["source_plan_task_cid"] == OBJECTIVE_REVISION, "objective revision drift")
    return add_projection_identity(
        payload, cid_field="revision_set_cid", sha_field="revision_set_sha256"
    )


def build_plan_admission_receipt(
    root: Mapping[str, Any], revisions: Mapping[str, Any]
) -> dict[str, Any]:
    def rejection(
        *,
        code: str,
        domain: str,
        message: str,
        source_ids: Sequence[str],
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = {
            "schema": "ipfs_accelerate_py/agent-supervisor/plan-admission-rejection@1",
            "code": code,
            "domain": domain,
            "message": message,
            "action_id": "",
            "effect_id": "",
            "dependency_id": "",
            "obligation_id": "",
            "source_ids": sorted(set(source_ids)),
            "details": dict(details),
        }
        return {**body, "rejection_id": supervisor_identity("plan-admission-rejection", body)}

    gates = root["gates"]
    rejections = sorted(
        (
            rejection(
                code="assumption_unresolved",
                domain="assumption",
                message="No source row has rights authority for learned training under this freeze.",
                source_ids=(root["root_cid"], root["bindings"]["rights"]["binding_cid"]),
                details={
                    "training_admitted_rows": 0,
                    "quarantined_source_count": 7173,
                },
            ),
            rejection(
                code="validation_failed",
                domain="validation",
                message="Rights, corpus, holdout, tokenizer, current-baseline, integrated-evidence, and portability gates are not all passed.",
                source_ids=(
                    root["root_cid"],
                    root["bindings"]["split"]["binding_cid"],
                    root["bindings"]["tokenizer_policy"]["binding_cid"],
                ),
                details={
                    "failed_gates": sorted(name for name, gate in gates.items() if not gate["passed"]),
                    "insufficient_holdouts": root["qualification"]["insufficient_holdouts"],
                    "portability_status": gates["portability"]["status"],
                    "semantic_baseline_currently_qualified": False,
                },
            ),
        ),
        key=lambda item: item["rejection_id"],
    )
    request_projection = {
        "campaign_input_root_cid": root["root_cid"],
        "candidate_plan_id": revisions["revision_set_cid"],
        "candidate_graph_id": revisions["candidate_graph_cid"],
        "repository_tree_id": root["repository"]["source_tree_id"],
    }
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/plan-admission-receipt@1",
        "compiler_version": 1,
        "requirement_id": IR_CONFORMANCE_REQUIREMENT_ID,
        "request_id": supervisor_identity("pgir-campaign-plan-admission-request", request_projection),
        "candidate_plan_id": revisions["revision_set_cid"],
        "candidate_graph_id": revisions["candidate_graph_cid"],
        "repository_tree_id": root["repository"]["source_tree_id"],
        "verdict": "rejected",
        "admitted": False,
        "semantic_roots": {
            "campaign": root["root_cid"],
            "compiler": root["bindings"]["compiler"]["binding_cid"],
            "corpus": root["bindings"]["corpus"]["binding_cid"],
            "decompiler": root["bindings"]["decompiler"]["binding_cid"],
            "examples": root["bindings"]["example_contracts"]["binding_cid"],
            "policy": root["bindings"]["policy"]["binding_cid"],
            "schema": root["bindings"]["schema_registry"]["binding_cid"],
            "source": root["bindings"]["source_snapshots"]["binding_cid"],
            "split": root["bindings"]["split"]["binding_cid"],
            "tokenizer_policy": root["bindings"]["tokenizer_policy"]["binding_cid"],
        },
        "intent_result_id": "",
        "legal_result_ids": [],
        "legal_permission_ids": [],
        "security_decision_ids": [],
        "security_grant_ids": [],
        "checked_dependency_ids": sorted(
            f"dependency:{item['task_id']}" for item in revisions["revisions"]
        ),
        "checked_assumption_ids": [
            "assumption:all-required-identities-resolved",
            "assumption:training-corpus-admitted",
        ],
        "generated_formula_ids": [],
        "proof_result_ids": [],
        "checked_validation_ids": [
            "validation:current-baseline",
            "validation:integrated-evidence",
            "validation:lineage-leakage",
            "validation:portability",
            "validation:referential-integrity",
            "validation:rights-admission",
            "validation:tokenizer-admission",
        ],
        "cve_security_evidence_ids": [],
        "rejection_reasons": rejections,
        "reason_codes": sorted({item["code"] for item in rejections}),
        "counterexamples": [],
        "local_replan_action_ids": [
            "replan:admit-rights-qualified-corpus",
            "replan:complete-required-holdouts",
            "replan:publish-or-document-portability-no-go",
            "replan:qualify-current-semantic-baseline",
            "replan:supersede-campaign-freeze-root",
        ],
        "closure_id": "",
        "permissions_are_grants": False,
        "generated_formulas_are_proofs": False,
        "authorizes_execution": False,
    }
    payload["receipt_id"] = supervisor_identity("plan-admission-receipt", payload)
    return payload


def build_verification_receipt(
    root: Mapping[str, Any],
    revisions: Mapping[str, Any],
    admission: Mapping[str, Any],
    verifier_run: Mapping[str, Any],
    portability: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        {"check_id": "previous-root-cid", "status": "passed", "evidence": HISTORICAL_ROOT_CID},
        {"check_id": "canonical-root-identity", "status": "passed", "evidence": root["root_cid"]},
        {"check_id": "pgir-211-result", "status": "passed", "evidence": P211_CID},
        {
            "check_id": "pgir-211-later-forest",
            "status": "passed",
            "evidence": root["integrated_evidence"]["later_forest"]["outer_commits"][2]["commit"],
        },
        {
            "check_id": "fresh-pgir-211-verifier-run",
            "status": "passed",
            "evidence": verifier_run["run_cid"],
        },
        {
            "check_id": "rights-corpus-holdout-tokenizer-baseline-integrated-portability-gates",
            "status": "passed",
            "evidence": "all seven gates evaluated; none authorize learned execution",
        },
        {
            "check_id": "portability-documented-no-go",
            "status": "passed",
            "evidence": portability["receipt_cid"],
        },
        {
            "check_id": "descendant-task-revisions",
            "status": "passed",
            "evidence": revisions["revision_set_cid"],
        },
        {
            "check_id": "supervisor-plan-admission",
            "status": "passed",
            "evidence": admission["receipt_id"],
        },
        {
            "check_id": "fail-closed-lease-barrier",
            "status": "passed",
            "evidence": "no-go root revises 2 descendants with zero lease-eligible tasks",
        },
    ]
    payload = {
        "schema": VERIFICATION_RECEIPT_SCHEMA,
        "verifier_interface": "pgir-successor-freeze-independent-verifier/v1",
        "campaign_input_root_cid": root["root_cid"],
        "revision_set_cid": revisions["revision_set_cid"],
        "plan_admission_receipt_id": admission["receipt_id"],
        "campaign_decision": root["qualification"]["decision"],
        "verification_verdict": "verified",
        "all_integrity_checks_passed": True,
        "authorizes_execution": False,
        "checks": checks,
    }
    return add_projection_identity(payload, cid_field="receipt_cid")


def load_inputs() -> dict[str, Any]:
    historical = strict_json(HISTORICAL_FREEZE / "campaign_input_root.json")
    verify_historical_root(historical)
    rights = strict_json(
        DATASETS_ROOT / "data/ir_learning/corpora/successor-v1/rights_manifest.json"
    )
    corpus = strict_json(
        DATASETS_ROOT / "data/ir_learning/corpora/successor-v1/corpus_root.json"
    )
    lineage = strict_json(
        DATASETS_ROOT / "data/ir_learning/corpora/successor-v1/lineage_graph.json"
    )
    split = strict_json(DATASETS_ROOT / "data/ir_learning/splits/successor-v1/split_root.json")
    holdouts = strict_json(
        DATASETS_ROOT / "data/ir_learning/splits/successor-v1/holdout_report.json"
    )
    tokenizer = strict_json(FREEZE_DIR / "tokenizer/tokenizer_policy.json")
    tokenizer_result = strict_json(FREEZE_DIR / "tokenizer/result.json")
    retirement = strict_json(
        DATASETS_ROOT
        / "data/ir_learning/evaluations/deterministic/successor-v1/retirement_receipt.json"
    )
    baseline = strict_json(FREEZE_DIR / "baseline-acceptance/baseline_acceptance.json")
    integrated = strict_json(FREEZE_DIR / "integrated-acceptance/integrated_acceptance.json")
    require(integrated["acceptance_cid"] == P211_CID, "loaded PGIR-211 CID drift")
    require(integrated["pgir_205_execution_authorized"] is False, "loaded PGIR-211 authorized execution")
    require(baseline["pgir_205_execution_authorized"] is False, "loaded PGIR-209 authorized execution")
    return {
        "historical": historical,
        "rights": rights,
        "corpus": corpus,
        "lineage": lineage,
        "split": split,
        "holdouts": holdouts,
        "tokenizer": tokenizer,
        "tokenizer_result": tokenizer_result,
        "retirement": retirement,
        "baseline": baseline,
        "integrated": integrated,
    }


def build_documents(*, issue_fresh: bool) -> dict[str, bytes]:
    inputs = load_inputs()
    repository = current_repository()
    forest = pgir_211_forest()
    if issue_fresh:
        require(
            git("rev-parse", "HEAD") == PGIR_211_COMPLETION,
            "fresh PGIR-205 capture must be issued at the exact PGIR-211 completion baseline",
        )
        portability = portability_probe(forest)
        verifier_run = run_pgir_211_verifier()
    else:
        portability = load_sealed_capture(
            "portability_receipt.json",
            cid_field="receipt_cid",
            sha_field="receipt_sha256",
        )
        verifier_run = load_sealed_capture(
            "pgir_211_baseline_verifier_run.json",
            cid_field="run_cid",
            sha_field="run_sha256",
        )
    validate_portability_receipt(portability)
    validate_pgir_211_verifier_run(verifier_run)
    gates = evaluate_gates(
        rights=inputs["rights"],
        corpus=inputs["corpus"],
        split=inputs["split"],
        holdouts=inputs["holdouts"],
        tokenizer=inputs["tokenizer"],
        tokenizer_result=inputs["tokenizer_result"],
        retirement=inputs["retirement"],
        baseline=inputs["baseline"],
        integrated=inputs["integrated"],
        verifier_run=verifier_run,
        portability=portability,
    )
    root = build_campaign_root(
        historical=inputs["historical"],
        rights=inputs["rights"],
        corpus=inputs["corpus"],
        lineage=inputs["lineage"],
        split=inputs["split"],
        holdouts=inputs["holdouts"],
        tokenizer=inputs["tokenizer"],
        tokenizer_result=inputs["tokenizer_result"],
        retirement=inputs["retirement"],
        baseline=inputs["baseline"],
        integrated=inputs["integrated"],
        forest=forest,
        verifier_run=verifier_run,
        portability=portability,
        gates=gates,
        repository=repository,
    )
    revisions = build_descendant_revisions(root)
    admission = build_plan_admission_receipt(root, revisions)
    verification = build_verification_receipt(
        root, revisions, admission, verifier_run, portability
    )
    documents: dict[str, bytes] = {
        "campaign_input_root.json": rendered_bytes(root),
        "descendant_task_revisions.json": rendered_bytes(revisions),
        "plan_admission_receipt.json": rendered_bytes(admission),
        "pgir_211_baseline_verifier_run.json": rendered_bytes(verifier_run),
        "portability_receipt.json": rendered_bytes(portability),
        "verification_receipt.json": rendered_bytes(verification),
    }
    static_names = (
        "README.md",
        "build_campaign_freeze.py",
        "ir_campaign_input_root.schema.json",
        "verify_campaign_freeze.py",
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for name in (*static_names, *sorted(documents)):
        data = documents.get(name)
        if data is None:
            path = FREEZE_DIR / name
            require(path.is_file() and not path.is_symlink(), f"required static freeze file absent: {name}")
            data = path.read_bytes()
        artifacts[name] = {
            "raw_cid": raw_cid(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    manifest = add_projection_identity(
        {
            "schema": MANIFEST_SCHEMA,
            "task_id": "PGIR-205",
            "campaign_input_root_cid": root["root_cid"],
            "revision_set_cid": revisions["revision_set_cid"],
            "plan_admission_receipt_id": admission["receipt_id"],
            "verification_receipt_cid": verification["receipt_cid"],
            "previous_root_cid": HISTORICAL_ROOT_CID,
            "pgir_211_acceptance_cid": P211_CID,
            "pgir_211_verifier_run_cid": verifier_run["run_cid"],
            "portability_receipt_cid": portability["receipt_cid"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "immutability": "supersede_never_overwrite",
            "decision": "no_go",
            "descendant_execution_authorized": False,
        },
        cid_field="manifest_cid",
    )
    documents["manifest.json"] = rendered_bytes(manifest)
    result = add_projection_identity(
        {
            "schema": RESULT_SCHEMA,
            "task_id": "PGIR-205",
            "objective_revision": OBJECTIVE_REVISION,
            "repository_id": REPOSITORY_ID,
            "source_tree_id": repository["source_tree_id"],
            "result_identity": "RESULT(PGIR-205)",
            "campaign_input_root_cid": root["root_cid"],
            "manifest_cid": manifest["manifest_cid"],
            "revision_set_cid": revisions["revision_set_cid"],
            "plan_admission_receipt_id": admission["receipt_id"],
            "verification_receipt_cid": verification["receipt_cid"],
            "previous_root_cid": HISTORICAL_ROOT_CID,
            "pgir_211_acceptance_cid": P211_CID,
            "pgir_211_verifier_run_cid": verifier_run["run_cid"],
            "portability_receipt_cid": portability["receipt_cid"],
            "disposition": "frozen_no_go",
            "decision": "no_go",
            "completion_authoritative": False,
            "descendant_execution_authorized": False,
            "training_task_eligible_count": 0,
            "unresolved_identities": [],
            "reason_codes": root["qualification"]["reason_codes"],
            "portability": portability["status"],
            "rollback": "retain this immutable successor root and create a separately admitted superseding root",
        },
        cid_field="result_cid",
        sha_field="result_sha256",
    )
    documents["result.json"] = rendered_bytes(result)
    return documents


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        require(path.is_file() and not path.is_symlink(), f"refusing unsafe existing output {path}")
        if path.read_bytes() != data:
            raise FreezeBuildError(
                f"immutable output differs at {path}; create a superseding freeze instead"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def check_documents(documents: Mapping[str, bytes]) -> None:
    missing: list[str] = []
    drifted: list[str] = []
    for name, expected in documents.items():
        path = FREEZE_DIR / name
        if not path.is_file() or path.is_symlink():
            missing.append(name)
        elif path.read_bytes() != expected:
            drifted.append(name)
    if missing or drifted:
        raise FreezeBuildError(
            f"freeze outputs do not match deterministic build; missing={missing}, drifted={drifted}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="create absent write-once JSON artifacts",
    )
    args = parser.parse_args(argv)
    capture_paths = (
        FREEZE_DIR / "pgir_211_baseline_verifier_run.json",
        FREEZE_DIR / "portability_receipt.json",
    )
    capture_presence = tuple(path.exists() for path in capture_paths)
    require(
        not any(capture_presence) or all(capture_presence),
        "sealed capture population is partial",
    )
    issue_fresh = bool(args.initialize and not all(capture_presence))
    documents = build_documents(issue_fresh=issue_fresh)
    if args.initialize:
        for name, data in documents.items():
            write_once(FREEZE_DIR / name, data)
        print("PGIR-205 successor freeze initialized or already exact")
        return 0
    check_documents(documents)
    print("PGIR-205 successor freeze verifies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
