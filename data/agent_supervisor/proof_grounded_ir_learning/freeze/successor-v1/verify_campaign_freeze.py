#!/usr/bin/env python3
"""Independently verify the immutable PGIR-205 successor-v1 freeze.

This verifier uses only the Python standard library.  It does not import the
builder or supervisor identity code.  A successful exit means the frozen
bytes, historical previous_root_cid, PGIR-211 result and later forest, fresh
verifier run, seven fail-closed gates, descendant revisions, and typed
rejection receipt all replayed.  It never turns a verified no-go into
execution authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

FREEZE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FREEZE_DIR.parents[4]
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"
HISTORICAL_ROOT = (
    REPOSITORY_ROOT
    / "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json"
)
PGIR_211_VERIFIER = (
    REPOSITORY_ROOT
    / "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py"
)

HISTORICAL_ROOT_CID = "baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q"
HISTORICAL_ROOT_SHA256 = "sha256:8a8cfcf2ebf7b735bcaf401299dd7806c3dff970c9bb81fdb199e0696a2cf575"
P208_CID = "baguqeeraburgmpdfo6weea57zlgkmppv7r34v2v3zstrepudxhrj6zrlgabq"
P209_CID = "baguqeerauh6r5lk47ecfmu5zjujmadrjiohd2ixkczcnurc33izkkvf2nb7q"
P210_CID = "baguqeera4ruaxwivpst2iwslorrgmbpuva6jqxczyjf6uditxg62atltnvkq"
P211_CID = "baguqeeram562re6snweb5nuinwprb4ehccvkin7kpylihktizlirsss7pllq"
PGIR_211_IMPLEMENTATION = "59cdab5572cac092fb42398fe908a424e54d9c4e"
PGIR_211_MERGE = "d48759e84dadff1d0dec2e43ee8ad19c534682d7"
PGIR_211_COMPLETION = "20ef9e48d59b505b04e6236a9a31aaba287c36fd"
PGIR_211_COMPLETION_TREE = "dcc71e6806dc12d32b0ae5f7655a60572c40ec30"
PGIR_211_TARGET = "75791d58beeab140c2a3ebaf9789705b3e75c151"
NESTED_CURRENT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
NESTED_CURRENT_TREE = "7169c2a67929044a02350bc26d0a51c853a4981b"
REPOSITORY_ID = "repository:sha256:3df67b4e7399635ecc20dc65888405eda8c32c7c28053e691fce8aa2aacaff4b"
OBJECTIVE_REVISION = "baguqeeraryux6dv5yim2by7j7zeffinnw5kfbl7prfiuew2yk6a2uknhnuiq"
SUCCESSOR_BOARD_PATH = "docs/architecture/proof_grounded_ir_learning/successor.todo.md"
SUCCESSOR_BOARD_BLOB = "fd110a44a8b530ba762bf2a702a77023514c2cb2"
TASK_IDENTITY_PATH = "ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py"
TASK_IDENTITY_BLOB = "5199029eae43a471591dadc62f59a94336ebc866"
TASK_IDENTITY_SHA256 = "sha256:4168c617ba079ca8fbeb2346931d4c7fcfb6516a09e5ddc7079434fb3541654e"
P203_POLICY_CID = "baguqeeraluqwxtejicycax65cicqfibtyy5kxkxxyng5nzbxmuooqedw2vka"
P203_RESULT_CID = "baguqeera2wxw5woodrk5534uu6wzwhxu7e3c3glizd4ns6bofd32qx43yhbq"
P204_RETIREMENT_CID = "baguqeeraw4nh2c7xxamku4juzt5257krzlzuaxe64vl5cuz4h4c4iwm6xdjq"
IR_CONFORMANCE_REQUIREMENT_ID = "287667496524558776121661391058779883318"
HIDDEN_TEST_COMMITMENT = (
    "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
)
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
REQUIRED_JSON = (
    "campaign_input_root.json",
    "descendant_task_revisions.json",
    "plan_admission_receipt.json",
    "pgir_211_baseline_verifier_run.json",
    "portability_receipt.json",
    "verification_receipt.json",
    "manifest.json",
    "result.json",
)
TOP_LEVEL_FILES = frozenset(
    {
        "README.md",
        "build_campaign_freeze.py",
        "ir_campaign_input_root.schema.json",
        "verify_campaign_freeze.py",
        *REQUIRED_JSON,
    }
)
MANIFEST_ARTIFACTS = frozenset(
    {
        "README.md",
        "build_campaign_freeze.py",
        "ir_campaign_input_root.schema.json",
        "verify_campaign_freeze.py",
        "campaign_input_root.json",
        "descendant_task_revisions.json",
        "plan_admission_receipt.json",
        "pgir_211_baseline_verifier_run.json",
        "portability_receipt.json",
        "verification_receipt.json",
    }
)
INHERITED_BINDING_NAMES = frozenset(
    {"compiler", "decompiler", "example_contracts", "gap_matrix", "schema_registry", "source_snapshots", "policy"}
)
CURRENT_TASK_IDENTITIES = {
    "PGIR-205": (
        "task/v1/8e297f0ebdc219a0e3e9fe4852a1adb75450afef8951425b585781aa29a76d11",
        OBJECTIVE_REVISION,
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
FULL_FILE_BINDINGS = {
    "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/rights_manifest.json": ("ipfs_datasets_py", NESTED_CURRENT),
    "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json": ("ipfs_datasets_py", NESTED_CURRENT),
    "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/lineage_graph.json": ("ipfs_datasets_py", NESTED_CURRENT),
    "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json": ("ipfs_datasets_py", NESTED_CURRENT),
    "ipfs_datasets_py/data/ir_learning/splits/successor-v1/holdout_report.json": ("ipfs_datasets_py", NESTED_CURRENT),
    "ipfs_datasets_py/data/ir_learning/evaluations/deterministic/successor-v1/retirement_receipt.json": ("ipfs_datasets_py", NESTED_CURRENT),
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/tokenizer_policy.json": ("ipfs_accelerate_py", PGIR_211_COMPLETION),
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/result.json": ("ipfs_accelerate_py", PGIR_211_COMPLETION),
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/baseline-acceptance/baseline_acceptance.json": ("ipfs_accelerate_py", PGIR_211_COMPLETION),
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/integrated_acceptance.json": ("ipfs_accelerate_py", PGIR_211_COMPLETION),
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json": ("ipfs_accelerate_py", PGIR_211_COMPLETION),
}
INHERITED_PORTABILITY_MISSING = (
    "04fbb09b4a8b34e77d11bd8da6642e0978baa02c",
    "597a0285738c5878eed462593fd75e18715ff7f8",
)
GATE_NAMES = (
    "rights",
    "corpus",
    "holdouts",
    "tokenizer",
    "current_baseline",
    "integrated_evidence",
    "portability",
)
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


class FreezeVerificationError(ValueError):
    """Raised when any identity, reference, or fail-closed gate drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeVerificationError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise FreezeVerificationError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise FreezeVerificationError(f"{path} contains unsupported {type(value).__name__}")


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


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or absent JSON file: {path}")

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
                FreezeVerificationError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                FreezeVerificationError(f"non-finite number {raw!r} in {path}")
            ),
        )
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    validate_value(value)
    return value


def verify_rendering(path: Path, value: Mapping[str, Any]) -> None:
    require(path.read_bytes() == rendered_bytes(value), f"noncanonical JSON rendering: {path}")


def _schema_matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise FreezeVerificationError(f"unsupported schema type {expected!r}")


def validate_schema_instance(
    value: Any, schema: Mapping[str, Any], root_schema: Mapping[str, Any], path: str = "$"
) -> None:
    if "$ref" in schema:
        ref = schema["$ref"]
        require(isinstance(ref, str) and ref.startswith("#/"), f"external schema ref at {path}")
        target: Any = root_schema
        for token in ref[2:].split("/"):
            target = target[token.replace("~1", "/").replace("~0", "~")]
        validate_schema_instance(value, target, root_schema, path)
        return
    if "oneOf" in schema:
        matches = 0
        for branch in schema["oneOf"]:
            try:
                validate_schema_instance(value, branch, root_schema, path)
            except FreezeVerificationError:
                continue
            matches += 1
        require(matches == 1, f"{path} matched {matches} oneOf branches")
    if "const" in schema:
        require(canonical_bytes(value) == canonical_bytes(schema["const"]), f"{path} const drift")
    if "enum" in schema:
        require(any(canonical_bytes(value) == canonical_bytes(item) for item in schema["enum"]), f"{path} enum drift")
    expected_type = schema.get("type")
    if expected_type is not None:
        types = [expected_type] if isinstance(expected_type, str) else expected_type
        require(any(_schema_matches_type(value, item) for item in types), f"{path} type drift")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            require(key in value, f"{path} missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(properties), f"{path} has unknown properties {sorted(set(value) - set(properties))}")
        for key, child in value.items():
            if key in properties:
                validate_schema_instance(child, properties[key], root_schema, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema:
            require(len(value) >= schema["minItems"], f"{path} has too few items")
        if "maxItems" in schema:
            require(len(value) <= schema["maxItems"], f"{path} has too many items")
        if schema.get("uniqueItems"):
            require(len({canonical_bytes(item) for item in value}) == len(value), f"{path} has duplicate items")
        if "items" in schema:
            for index, child in enumerate(value):
                validate_schema_instance(child, schema["items"], root_schema, f"{path}[{index}]")
    if isinstance(value, str):
        if "minLength" in schema:
            require(len(value) >= schema["minLength"], f"{path} is too short")
        if "pattern" in schema:
            require(re.search(schema["pattern"], value) is not None, f"{path} pattern drift")
    if isinstance(value, int) and not isinstance(value, bool) and "minimum" in schema:
        require(value >= schema["minimum"], f"{path} is below minimum")


def verify_root_schema(root: Mapping[str, Any]) -> None:
    schema_path = FREEZE_DIR / "ir_campaign_input_root.schema.json"
    schema = strict_json(schema_path)
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "schema draft drift")
    require(schema.get("$id") == "https://schemas.ipfs-accelerate.local/agent-supervisor/ir-campaign-input-root-v2.json", "schema ID drift")
    require(schema.get("additionalProperties") is False, "root schema is open")

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                require(node.get("additionalProperties") is False, f"open object schema at {path}")
            for key, child in node.items():
                walk(child, f"{path}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}/{index}")

    walk(schema, "$")
    validate_schema_instance(root, schema, schema)


def verify_projection_identity(
    value: Mapping[str, Any], *, cid_field: str, sha_field: str | None = None
) -> None:
    require(cid_field in value, f"missing {cid_field}")
    projection = dict(value)
    claimed_cid = projection.pop(cid_field)
    claimed_sha = projection.pop(sha_field) if sha_field else None
    require(claimed_cid == dag_json_cid(projection), f"{cid_field} does not match projection")
    if sha_field:
        expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
        require(claimed_sha == expected_sha, f"{sha_field} does not match projection")


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
        raise FreezeVerificationError(
            f"git {' '.join(args)} failed in {cwd}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def git_bytes(*args: str, cwd: Path = REPOSITORY_ROOT) -> bytes:
    process = subprocess.run(
        ("git", *args), cwd=cwd, check=False, capture_output=True, env=VERIFIER_ENVIRONMENT
    )
    if process.returncode:
        raise FreezeVerificationError(
            f"git {' '.join(args)} failed in {cwd}: {process.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return process.stdout


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def index_blob(path: str, *, repository: str) -> str | None:
    cwd = DATASETS_ROOT if repository == "ipfs_datasets_py" else REPOSITORY_ROOT
    object_path = path.removeprefix("ipfs_datasets_py/") if repository == "ipfs_datasets_py" else path
    rows = git("ls-files", "--stage", "--", object_path, cwd=cwd).splitlines()
    if not rows:
        return None
    require(len(rows) == 1, f"non-unique index entry for {path}")
    match = re.fullmatch(r"100\d{3} ([0-9a-f]{40}) 0\t.+", rows[0])
    require(match is not None, f"unsafe index stage or mode for {path}")
    return match.group(1)


def gitlink(commit: str) -> str:
    line = git("ls-tree", commit, "ipfs_datasets_py")
    parts = line.split()
    require(len(parts) >= 3 and parts[0] == "160000", f"missing gitlink at {commit}")
    return parts[2]


def verify_file_binding(binding: Mapping[str, Any]) -> bytes:
    require(
        set(binding) == {"path", "repository", "raw_cid", "sha256", "size_bytes", "git_blob", "revision"},
        f"file-binding field population drift: {binding.get('path')}",
    )
    relative = str(binding["path"])
    require(relative and not relative.startswith("/") and ".." not in Path(relative).parts, f"unsafe bound path: {relative}")
    require(relative in FULL_FILE_BINDINGS, f"unexpected full file binding: {relative}")
    expected_repository, expected_revision = FULL_FILE_BINDINGS[relative]
    require(binding["repository"] == expected_repository, f"repository drift: {relative}")
    require(binding["revision"] == expected_revision, f"revision drift: {relative}")
    path = REPOSITORY_ROOT / relative
    require(path.is_file() and not path.is_symlink(), f"bound file absent: {binding['path']}")
    data = path.read_bytes()
    require(binding["size_bytes"] == len(data), f"size drift: {binding['path']}")
    require(
        binding["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(),
        f"sha256 drift: {binding['path']}",
    )
    require(binding["raw_cid"] == raw_cid(data), f"raw CID drift: {binding['path']}")
    repository = binding["repository"]
    cwd = DATASETS_ROOT if repository == "ipfs_datasets_py" else REPOSITORY_ROOT
    object_path = relative.removeprefix("ipfs_datasets_py/") if repository == "ipfs_datasets_py" else relative
    require(git_blob_oid(data) == binding["git_blob"], f"computed Git blob drift: {relative}")
    require(
        git("rev-parse", f"{binding['revision']}:{object_path}", cwd=cwd) == binding["git_blob"],
        f"revision Git blob drift: {relative}",
    )
    require(git_bytes("cat-file", "blob", binding["git_blob"], cwd=cwd) == data, f"Git object bytes drift: {relative}")
    require(index_blob(relative, repository=repository) == binding["git_blob"], f"index blob drift: {relative}")
    return data


def verify_recursive_file_bindings(root: Mapping[str, Any]) -> None:
    found: dict[str, Mapping[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "revision" in value and {"path", "repository", "raw_cid", "sha256", "size_bytes", "git_blob"} <= set(value):
                verify_file_binding(value)
                previous = found.setdefault(value["path"], value)
                require(previous == value, f"inconsistent duplicate binding: {value['path']}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(root)
    require(set(found) == set(FULL_FILE_BINDINGS), f"full file-binding path population drift: {sorted(set(found) ^ set(FULL_FILE_BINDINGS))}")


def verify_legacy_file_bindings(bindings: Mapping[str, Any]) -> None:
    expected_keys = {"path", "repository", "raw_cid", "sha256", "size_bytes", "git_blob"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == expected_keys:
                relative = value["path"]
                repository = value["repository"]
                require(repository in {"ipfs_accelerate_py", "ipfs_datasets_py"}, f"legacy repository drift: {relative}")
                cwd = DATASETS_ROOT if repository == "ipfs_datasets_py" else REPOSITORY_ROOT
                # IRCampaignInputRoot@1 records predate an explicit revision.
                # Replay their immutable Git objects, never current descendant
                # disk/index bytes, which may legitimately supersede them.
                data = git_bytes("cat-file", "blob", value["git_blob"], cwd=cwd)
                require(len(data) == value["size_bytes"], f"legacy size drift: {relative}")
                require(raw_cid(data) == value["raw_cid"], f"legacy raw CID drift: {relative}")
                require("sha256:" + hashlib.sha256(data).hexdigest() == value["sha256"], f"legacy hash drift: {relative}")
                require(git_blob_oid(data) == value["git_blob"], f"legacy Git blob drift: {relative}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(bindings)


def verify_forest(forest: Mapping[str, Any]) -> None:
    commits = (
        (
            "implementation",
            PGIR_211_IMPLEMENTATION,
            "8e44a1024b9152cc988be0a9dad0fada2f0eecf4",
            ["10c38f56803442c224d379cedf6b0e5ca1e35147"],
            "feat(pgir): seal integrated successor acceptance",
            "benjamin barber <starworks5@gmail.com>",
        ),
        (
            "merge",
            PGIR_211_MERGE,
            "d6f82932e195d5bd885322d0988696023e4eeb22",
            ["8cbb26404298a6f0b34e65c363444beb075e3dbe", PGIR_211_IMPLEMENTATION],
            "Merge commit '59cdab5572cac092fb42398fe908a424e54d9c4e' into agent/pgir-successor-current-supervisor-20260825",
            "benjamin barber <starworks5@gmail.com>",
        ),
        (
            "completion",
            PGIR_211_COMPLETION,
            PGIR_211_COMPLETION_TREE,
            [PGIR_211_MERGE],
            "PGIR-211: mark todo completed",
            "Implementation Daemon <implementation-daemon@example.invalid>",
        ),
    )
    expected_rows = []
    for role, commit, tree, parents, subject, identity in commits:
        require(git("cat-file", "-t", commit) == "commit", f"{role} is not a commit")
        require(git("rev-parse", f"{commit}^{{tree}}") == tree, f"{role} tree drift")
        require(git("show", "-s", "--format=%P", commit).split() == parents, f"{role} parent drift")
        require(git("show", "-s", "--format=%s", commit) == subject, f"{role} subject drift")
        require(git("show", "-s", "--format=%an <%ae>", commit) == identity, f"{role} author drift")
        require(git("show", "-s", "--format=%cn <%ce>", commit) == identity, f"{role} committer drift")
        require(gitlink(commit) == NESTED_CURRENT, f"{role} gitlink drift")
        require(git("merge-base", "--is-ancestor", commit, PGIR_211_COMPLETION) == "", f"{role} is not an ancestor of Q")
        expected_rows.append(
            {"task_id": "PGIR-211", "role": role, "commit": commit, "tree": tree, "parents": parents, "subject": subject, "gitlink": NESTED_CURRENT}
        )
    expected_forest = {
        "schema": "proof-grounded-ir-learning/pgir-211-later-forest@1",
        "result_identity": "RESULT(PGIR-211)",
        "containing_commit_claimed": False,
        "circular_self_reference_avoided": True,
        "integrated_target": {"commit": PGIR_211_TARGET, "tree": "e092bc48487226229c0df5c47029c3db36004e18", "gitlink": NESTED_CURRENT},
        "outer_commits": expected_rows,
        "nested": {"commit": NESTED_CURRENT, "tree": NESTED_CURRENT_TREE, "parents": ["8736a0023d5d3afe4d0e5b044a3e4480966a8bf7"]},
        "note": "RESULT(PGIR-211) cannot name this later forest; PGIR-205 binds it at the freeze baseline.",
    }
    require(forest == expected_forest, "exact PGIR-211 later-forest envelope drift")
    require(git("rev-parse", f"{PGIR_211_TARGET}^{{tree}}") == expected_forest["integrated_target"]["tree"], "integrated target tree drift")
    require(gitlink(PGIR_211_TARGET) == NESTED_CURRENT, "integrated target gitlink drift")
    require(git("merge-base", "--is-ancestor", PGIR_211_TARGET, PGIR_211_COMPLETION) == "", "integrated target is not an ancestor of Q")
    require(git("show", "-s", "--format=%T", NESTED_CURRENT, cwd=DATASETS_ROOT) == NESTED_CURRENT_TREE, "nested tree drift")
    require(git("show", "-s", "--format=%P", NESTED_CURRENT, cwd=DATASETS_ROOT).split() == expected_forest["nested"]["parents"], "nested parent drift")
    daemon = "Implementation Daemon <implementation-daemon@example.invalid>"
    require(git("show", "-s", "--format=%an <%ae>", NESTED_CURRENT, cwd=DATASETS_ROOT) == daemon, "nested author drift")
    require(git("show", "-s", "--format=%cn <%ce>", NESTED_CURRENT, cwd=DATASETS_ROOT) == daemon, "nested committer drift")
    require(git("show", "-s", "--format=%s", NESTED_CURRENT, cwd=DATASETS_ROOT) == "PGIR-204: Requalify or replace the historical R1 semantic baseline", "nested subject drift")

    integrated_paths = {
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/README.md",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/capture_evidence.py",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/component_verification_receipt.json",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/historical_closure_receipt.json",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/integrated_acceptance.json",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/network_receipt.json",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/portability_receipt.json",
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/test_receipt.json",
        "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py",
    }

    def changed(parent: str, commit: str) -> dict[str, str]:
        rows = git("diff-tree", "--no-commit-id", "--name-status", "-r", parent, commit).splitlines()
        parsed = [row.split("\t", 1) for row in rows if row]
        require(all(len(row) == 2 for row in parsed), f"rename/copy drift at {commit}")
        return {path: status for status, path in parsed}

    require(changed(commits[0][3][0], PGIR_211_IMPLEMENTATION) == {path: "A" for path in integrated_paths}, "implementation path population drift")
    require(changed(commits[1][3][0], PGIR_211_MERGE) == {path: "A" for path in integrated_paths}, "merge first-parent path population drift")
    require(
        changed(PGIR_211_IMPLEMENTATION, PGIR_211_MERGE)
        == {path: "M" for path in {
            SUCCESSOR_BOARD_PATH,
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
            "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
            "ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py",
            "test/api/test_agent_supervisor_direct_codex_isolation.py",
            "test/api/test_agent_supervisor_validation_scheduler.py",
        }},
        "merge second-parent path population drift",
    )
    require(changed(PGIR_211_MERGE, PGIR_211_COMPLETION) == {SUCCESSOR_BOARD_PATH: "M"}, "completion path population drift")
    require(
        git("merge-base", "--is-ancestor", PGIR_211_COMPLETION, "HEAD") == "",
        "current HEAD does not descend from the pinned freeze baseline",
    )
    require(git("rev-parse", "HEAD", cwd=DATASETS_ROOT) == NESTED_CURRENT, "nested HEAD drift")
    require(git("merge-base", "--is-ancestor", PGIR_211_TARGET, "HEAD") == "", "HEAD does not descend from RESULT(PGIR-211) target")


def verify_pgir_211_run(run: Mapping[str, Any], *, fresh_network: bool) -> None:
    verify_projection_identity(run, cid_field="run_cid", sha_field="run_sha256")
    require(
        set(run)
        == {
            "schema", "task_id", "result_identity", "acceptance_cid", "mode",
            "argv", "cwd", "exit_code", "started_at_utc", "ended_at_utc",
            "stdout", "stdout_raw_cid", "stdout_sha256", "stdout_size_bytes",
            "stderr_raw_cid", "stderr_sha256", "stderr_size_bytes", "stderr_empty",
            "pgir_205_execution_authorized", "baseline", "verifier_source",
            "environment", "note", "run_sha256", "run_cid",
        },
        "sealed PGIR-211 verifier-run field population drift",
    )
    require(
        run["schema"] == "proof-grounded-ir-learning/pgir-211-baseline-verifier-run@1",
        "verifier-run schema drift",
    )
    require(run["task_id"] == "PGIR-205", "verifier-run task drift")
    require(run["result_identity"] == "RESULT(PGIR-211)", "verifier-run result identity drift")
    require(run["acceptance_cid"] == P211_CID, "verifier-run acceptance CID drift")
    require(run["pgir_205_execution_authorized"] is False, "sealed verifier run authorized PGIR-205")
    require(run["stdout"].get("pgir_205_execution_authorized") is False, "fresh stdout authorized PGIR-205")
    require(run["stdout"].get("decision") == "permanent_no_go", "fresh stdout is not permanent_no_go")
    require(run["mode"] == "network", "sealed PGIR-211 run is not network mode")
    argv = [
        "/usr/bin/python3.12",
        "-S",
        "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py",
        "--network",
    ]
    require(run["argv"] == argv, "sealed PGIR-211 verifier argv drift")
    require(run["cwd"] == ".", "sealed verifier cwd drift")
    require(run["exit_code"] == 0, "sealed network verifier did not exit zero")
    require(run["stderr_empty"] is True, "sealed network verifier emitted stderr")
    require(run["stderr_size_bytes"] == 0, "sealed stderr size drift")
    require(run["stderr_raw_cid"] == raw_cid(b""), "sealed stderr CID drift")
    require(
        run["stderr_sha256"] == "sha256:" + hashlib.sha256(b"").hexdigest(),
        "sealed stderr hash drift",
    )
    expected_stdout = canonical_bytes(run["stdout"]) + b"\n"
    require(run["stdout_size_bytes"] == len(expected_stdout), "sealed stdout size drift")
    require(run["stdout_raw_cid"] == raw_cid(expected_stdout), "sealed stdout CID drift")
    require(
        run["stdout_sha256"] == "sha256:" + hashlib.sha256(expected_stdout).hexdigest(),
        "sealed stdout hash drift",
    )
    require(
        run["baseline"]
        == {
            "outer_commit": PGIR_211_COMPLETION,
            "outer_tree": PGIR_211_COMPLETION_TREE,
            "nested_gitlink": NESTED_CURRENT,
            "nested_tree": NESTED_CURRENT_TREE,
        },
        "sealed verifier baseline drift",
    )
    for field in ("started_at_utc", "ended_at_utc"):
        require(datetime.fromisoformat(run[field]).tzinfo is not None, f"{field} is not timezone-aware")
    require(
        datetime.fromisoformat(run["started_at_utc"])
        <= datetime.fromisoformat(run["ended_at_utc"]),
        "sealed verifier interval is reversed",
    )
    require(
        run["stdout"]
        == {
            "acceptance_cid": P211_CID,
            "completion_authoritative": False,
            "component_verified": True,
            "decision": "permanent_no_go",
            "live_network": {
                "matched": 21,
                "receipt_replay_used_as_substitute": False,
                "requested": 21,
                "transport": "curl --disable HTTPS exact-revision GET under exact minimal environment",
            },
            "pgir_205_execution_authorized": False,
            "verified": True,
        },
        "sealed network verifier stdout drift",
    )
    source = run["verifier_source"]
    require(set(source) == {"path", "revision", "git_blob", "raw_cid", "sha256", "size_bytes"}, "verifier source field population drift")
    require(
        source.get("path")
        == "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py"
        and source.get("revision") == PGIR_211_COMPLETION
        and source.get("git_blob") == "712be25b94e24cfa2e53a02140bb1885210103c5",
        "sealed verifier source drift",
    )
    require(PGIR_211_VERIFIER.is_file() and not PGIR_211_VERIFIER.is_symlink(), "unsafe verifier source path")
    source_data = PGIR_211_VERIFIER.read_bytes()
    require(source.get("size_bytes") == len(source_data), "verifier source size drift")
    require(source.get("raw_cid") == raw_cid(source_data), "verifier source CID drift")
    require(
        source.get("sha256") == "sha256:" + hashlib.sha256(source_data).hexdigest(),
        "verifier source hash drift",
    )
    require(
        git("rev-parse", f"{PGIR_211_COMPLETION}:{source['path']}") == source["git_blob"],
        "verifier source revision binding drift",
    )
    require(git_blob_oid(source_data) == source["git_blob"], "verifier source computed Git blob drift")
    require(git_bytes("cat-file", "blob", source["git_blob"]) == source_data, "verifier source Git object bytes drift")
    require(index_blob(source["path"], repository="ipfs_accelerate_py") == source["git_blob"], "verifier source index drift")
    require(
        run["environment"]
        == {"executable": "/usr/bin/python3.12", "no_site": True, "pythonpath": SAFE_PYTHONPATH, "path": "/usr/bin:/bin"},
        "sealed verifier environment drift",
    )
    require(
        run["note"]
        == "Fresh run at the PGIR-205 baseline; it does not circularly name the PGIR-211 completion commit as RESULT(PGIR-211).",
        "sealed verifier note drift",
    )
    if not fresh_network:
        return
    process = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        env=VERIFIER_ENVIRONMENT,
        timeout=1200,
    )
    require(process.returncode == run["exit_code"], "replayed PGIR-211 verifier exit drift")
    require(process.stderr == b"", "replayed PGIR-211 verifier emitted stderr")
    require(process.stdout == expected_stdout, "replayed PGIR-211 verifier raw stdout drift")
    replayed = json.loads(process.stdout.decode("utf-8"))
    require(replayed == run["stdout"], "replayed PGIR-211 verifier stdout drift")
    require(replayed.get("pgir_205_execution_authorized") is False, "replayed PGIR-211 run authorized PGIR-205")


def normalize_identity_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_identity_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text).rstrip("/")


def _mapping_value(source: Mapping[str, Any], *keys: str) -> Any:
    normalized = {str(key).strip().casefold().replace("_", " "): value for key, value in source.items()}
    for key in keys:
        candidate = normalized.get(key.casefold().replace("_", " "))
        if candidate not in (None, "", [], ()):
            return candidate
    return ""


def _sequence(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [item for item in value if item not in (None, "")]
    return [value]


def canonical_task_identity(task: Mapping[str, Any], *, semantic_key: str = "") -> dict[str, str]:
    """Independent mirror of the pinned current supervisor task identity API."""

    source = dict(task)
    metadata = dict(source.get("metadata") or {})
    if semantic_key:
        metadata["semantic key"] = semantic_key
    provided_key = str(_mapping_value(source, "canonical task key") or _mapping_value(metadata, "canonical task key") or "").strip()
    provided_cid = str(_mapping_value(source, "canonical task cid") or _mapping_value(metadata, "canonical task cid") or "").strip()
    allowed_paths = sorted(
        {
            path
            for item in _sequence(_mapping_value(source, "allowed paths") or _mapping_value(metadata, "allowed paths"))
            if (path := normalize_identity_path(item))
            and path != "."
            and not path.startswith("/")
            and ".." not in Path(path).parts
            and not any(marker in path for marker in "*?[")
        }
    )
    if provided_key and provided_cid:
        if not allowed_paths:
            suffix = provided_key.rsplit("/", 1)[-1].casefold()
            fingerprint = suffix if re.fullmatch(r"[0-9a-f]{64}", suffix) else hashlib.sha256(canonical_bytes([provided_key, provided_cid])).hexdigest()
            return {"canonical_task_key": provided_key, "canonical_task_cid": provided_cid, "semantic_fingerprint": fingerprint}
        material: dict[str, Any] = {
            "schema": "ipfs_accelerate_py/agent-supervisor/task-identity@1",
            "provided_identity": {"canonical_task_key": provided_key, "canonical_task_cid": provided_cid},
            "authority": {"additional_allowed_paths": allowed_paths},
        }
    else:
        explicit_key = normalize_identity_text(_mapping_value(source, "dedupe key") or _mapping_value(metadata, "dedupe key"))
        if explicit_key:
            material = {"schema": "ipfs_accelerate_py/agent-supervisor/task-identity@1", "explicit_key": explicit_key}
        else:
            semantic = {
                key: value
                for key, value in {
                    "title": normalize_identity_text(_mapping_value(source, "title", "summary") or _mapping_value(metadata, "title", "summary")),
                    "outputs": sorted({normalize_identity_path(item) for item in _sequence(_mapping_value(source, "outputs", "paths", "files") or _mapping_value(metadata, "outputs", "paths", "files")) if normalize_identity_path(item)}),
                    "acceptance": [normalize_identity_text(item) for item in _sequence(_mapping_value(source, "acceptance", "acceptance criteria") or _mapping_value(metadata, "acceptance", "acceptance criteria")) if normalize_identity_text(item)],
                    "evidence": sorted({normalize_identity_text(item) for item in _sequence(_mapping_value(source, "missing evidence", "evidence") or _mapping_value(metadata, "missing evidence", "evidence")) if normalize_identity_text(item)}),
                    "evidence_outputs": sorted({normalize_identity_path(item) for item in _sequence(_mapping_value(source, "evidence outputs") or _mapping_value(metadata, "evidence outputs")) if normalize_identity_path(item)}),
                    "goal": normalize_identity_text(_mapping_value(source, "goal id", "goal packet key", "goal") or _mapping_value(metadata, "goal id", "goal packet key", "goal")),
                    "semantic_hint": normalize_identity_text(_mapping_value(source, "semantic key", "bundle key", "work scope", "fingerprint") or _mapping_value(metadata, "semantic key", "bundle key", "work scope", "fingerprint")),
                }.items()
                if value
            }
            require(bool(semantic), "task identity has no semantic material")
            material = {"schema": "ipfs_accelerate_py/agent-supervisor/task-identity@1", "semantic": semantic}
        if allowed_paths:
            material["authority"] = {"additional_allowed_paths": allowed_paths}
    fingerprint = hashlib.sha256(canonical_bytes(material)).hexdigest()
    return {
        "canonical_task_key": f"task/v1/{fingerprint}",
        "canonical_task_cid": dag_json_cid(material),
        "semantic_fingerprint": fingerprint,
    }


def pinned_board_tasks() -> dict[str, dict[str, Any]]:
    require(git("rev-parse", f"{PGIR_211_COMPLETION}:{SUCCESSOR_BOARD_PATH}") == SUCCESSOR_BOARD_BLOB, "pinned board blob drift")
    board = git_bytes("cat-file", "blob", SUCCESSOR_BOARD_BLOB)
    parent_board = git_bytes("show", f"{PGIR_211_MERGE}:{SUCCESSOR_BOARD_PATH}")
    require(board.count(b"## PGIR-211 ") == 1 and parent_board.count(b"## PGIR-211 ") == 1, "PGIR-211 board section population drift")
    before = parent_board.decode("utf-8")
    after = board.decode("utf-8")
    require(before.replace("## PGIR-211 ", "\0## PGIR-211 ", 1).split("\0", 1)[1].split("\n## ", 1)[0].count("- Status: todo") == 1, "pre-Q PGIR-211 status drift")
    require(after.replace("## PGIR-211 ", "\0## PGIR-211 ", 1).split("\0", 1)[1].split("\n## ", 1)[0].count("- Status: completed") == 1, "Q PGIR-211 status drift")
    source = git_bytes("cat-file", "blob", TASK_IDENTITY_BLOB)
    require(git("rev-parse", f"{PGIR_211_COMPLETION}:{TASK_IDENTITY_PATH}") == TASK_IDENTITY_BLOB, "pinned task-identity blob drift")
    require("sha256:" + hashlib.sha256(source).hexdigest() == TASK_IDENTITY_SHA256, "pinned task-identity source hash drift")
    require((REPOSITORY_ROOT / TASK_IDENTITY_PATH).read_bytes() == source, "disk task-identity API differs from Q")
    require(index_blob(TASK_IDENTITY_PATH, repository="ipfs_accelerate_py") == TASK_IDENTITY_BLOB, "task-identity index binding drift")

    tasks: dict[str, dict[str, Any]] = {}
    current_id = ""
    title = ""
    lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, title, lines
        if not current_id:
            return
        metadata: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- ") and ":" in stripped:
                key, value = stripped[2:].split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        split_csv = lambda value: [item for raw in str(value or "").split(",") if (item := raw.strip()) and item.lower() not in {"none", "n/a"}]
        task = {
            "task_id": current_id,
            "title": title,
            "depends_on": split_csv(metadata.get("depends on", "")),
            "outputs": split_csv(metadata.get("outputs", "")),
            "acceptance": metadata.get("acceptance", ""),
            "metadata": metadata,
        }
        task.update(canonical_task_identity(task))
        require(current_id not in tasks, f"duplicate pinned board task {current_id}")
        tasks[current_id] = task
        current_id, title, lines = "", "", []

    for line in after.splitlines():
        if line.startswith("## "):
            flush()
            if line.startswith("## PGIR-"):
                current_id, _, title = line[3:].strip().partition(" ")
        elif current_id:
            lines.append(line)
    flush()
    return tasks


def verify_descendant_revisions(revisions: Mapping[str, Any], root: Mapping[str, Any]) -> None:
    tasks = pinned_board_tasks()
    for task_id, expected in CURRENT_TASK_IDENTITIES.items():
        actual = tasks[task_id]
        require((actual["canonical_task_key"], actual["canonical_task_cid"]) == expected, f"{task_id} current supervisor identity drift")
    semantic_key = f"pgir-campaign-input-root@2:{root['root_cid']}"
    expected_rows = []
    for task_id in ("PGIR-206", "PGIR-207"):
        task = tasks[task_id]
        revised = canonical_task_identity(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "outputs": task["outputs"],
                "acceptance": task["acceptance"],
                "metadata": task["metadata"],
            },
            semantic_key=semantic_key,
        )
        expected_rows.append(
            {
                "task_id": task_id,
                "title": task["title"],
                "depends_on": task["depends_on"],
                "current_task_cid": task["canonical_task_cid"],
                "current_task_key": task["canonical_task_key"],
                "revised_task_cid": revised["canonical_task_cid"],
                "revised_task_key": revised["canonical_task_key"],
                "semantic_fingerprint": revised["semantic_fingerprint"],
                "input_binding": {"campaign_input_root_cid": root["root_cid"], "semantic_key": semantic_key, "decision": "no_go"},
                "lease_eligible": False,
                "block_reason_codes": root["qualification"]["reason_codes"],
            }
        )
    graph = {
        "schema": "PGIRDescendantTaskGraph@1",
        "root_task_id": "PGIR-205",
        "tasks": [{"task_id": row["task_id"], "task_cid": row["revised_task_cid"], "depends_on": row["depends_on"]} for row in expected_rows],
    }
    expected_payload = {
        "schema": "PGIRDescendantTaskRevisionSet@1",
        "task_id": "PGIR-205",
        "campaign_input_root_cid": root["root_cid"],
        "source_plan_task_cid": OBJECTIVE_REVISION,
        "source_plan_task_key": CURRENT_TASK_IDENTITIES["PGIR-205"][0],
        "identity_source": {"board_revision": PGIR_211_COMPLETION, "board_blob": SUCCESSOR_BOARD_BLOB, "task_identity_revision": PGIR_211_COMPLETION, "task_identity_blob": TASK_IDENTITY_BLOB},
        "revision_patch": {"metadata_field": "Semantic key", "metadata_value": semantic_key, "protected_board_mutated": False, "application_policy": "supervisor_compare_and_swap_after_superseding_admission"},
        "candidate_graph_cid": dag_json_cid(graph),
        "descendant_task_count": 2,
        "lease_eligible_count": 0,
        "revisions": expected_rows,
    }
    expected = dict(expected_payload)
    expected["revision_set_sha256"] = "sha256:" + hashlib.sha256(canonical_bytes(expected_payload)).hexdigest()
    expected["revision_set_cid"] = dag_json_cid(expected_payload)
    require(revisions == expected, "descendant revision set differs from independent current task identities")


def verify_portability(portability: Mapping[str, Any]) -> None:
    require(
        set(portability)
        == {
            "schema", "task_id", "started_at_utc", "ended_at_utc", "baseline",
            "status", "observation_method", "inherited_pgir_211_missing_outer_commits",
            "later_forest_commits", "missing_outer_commits", "missing_nested_commits",
            "nested_current_remote_refs", "pgir_205_execution_authorized", "effect",
            "receipt_sha256", "receipt_cid",
        },
        "portability receipt field population drift",
    )
    verify_projection_identity(portability, cid_field="receipt_cid", sha_field="receipt_sha256")
    require(portability["schema"] == "proof-grounded-ir-learning/pgir-205-portability-receipt@1", "portability schema drift")
    require(portability["task_id"] == "PGIR-205", "portability task drift")
    require(
        portability["baseline"]
        == {"outer_commit": PGIR_211_COMPLETION, "outer_tree": PGIR_211_COMPLETION_TREE, "nested_gitlink": NESTED_CURRENT, "nested_tree": NESTED_CURRENT_TREE},
        "portability baseline drift",
    )
    require(portability["status"] == "portability_no_go", "portability status drift")
    require(portability["inherited_pgir_211_missing_outer_commits"] == list(INHERITED_PORTABILITY_MISSING), "inherited portability population drift")
    require(portability["later_forest_commits"] == [PGIR_211_IMPLEMENTATION, PGIR_211_MERGE, PGIR_211_COMPLETION], "later portability forest population drift")
    require(
        portability["missing_outer_commits"]
        == sorted({*INHERITED_PORTABILITY_MISSING, PGIR_211_IMPLEMENTATION, PGIR_211_MERGE, PGIR_211_COMPLETION}),
        "missing outer portability population drift",
    )
    require(portability["missing_nested_commits"] == [], "missing nested portability drift")
    require(portability["nested_current_remote_refs"] == ["refs/remotes/origin/HEAD", "refs/remotes/origin/main"], "nested remote-ref observation drift")
    require(
        portability["observation_method"]
        == "git for-each-ref --contains <commit> --format=%(refname) refs/remotes in the current outer and nested repositories; local stale remote-tracking refs are insufficient for a fresh recursive checkout",
        "portability observation method drift",
    )
    require(
        portability["effect"]
        == "PGIR-205 remains fail-closed: no materialization, training, promotion, or execution authority follows from this freeze.",
        "portability effect drift",
    )
    require(portability["pgir_205_execution_authorized"] is False, "portability authorized execution")
    start = datetime.fromisoformat(portability["started_at_utc"])
    end = datetime.fromisoformat(portability["ended_at_utc"])
    require(start.tzinfo is not None and end.tzinfo is not None and start <= end, "portability interval drift")


def verify_bundle_population(artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    population = {path.name for path in FREEZE_DIR.iterdir() if path.is_file() or path.is_symlink()}
    require(population == TOP_LEVEL_FILES, f"freeze top-level file population drift: {sorted(population ^ TOP_LEVEL_FILES)}")
    for name in REQUIRED_JSON:
        verify_rendering(FREEZE_DIR / name, artifacts[name])
    tracked: dict[str, str | None] = {}
    for name in TOP_LEVEL_FILES:
        relative = (FREEZE_DIR / name).relative_to(REPOSITORY_ROOT).as_posix()
        require(not (FREEZE_DIR / name).is_symlink(), f"freeze artifact is a symlink: {name}")
        tracked[name] = index_blob(relative, repository="ipfs_accelerate_py")
    require(all(value is None for value in tracked.values()) or all(value is not None for value in tracked.values()), "freeze bundle is only partially represented in the Git index")
    if all(value is not None for value in tracked.values()):
        for name, blob in tracked.items():
            data = (FREEZE_DIR / name).read_bytes()
            require(blob == git_blob_oid(data), f"freeze index bytes drift: {name}")
            require(git_bytes("cat-file", "blob", str(blob)) == data, f"freeze Git object bytes drift: {name}")


def verify_manifest_and_crosslinks(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Mapping[str, Any],
    revisions: Mapping[str, Any],
    admission: Mapping[str, Any],
    run: Mapping[str, Any],
    portability: Mapping[str, Any],
    verification: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    require(set(manifest) == {
        "schema", "task_id", "campaign_input_root_cid", "revision_set_cid",
        "plan_admission_receipt_id", "verification_receipt_cid", "previous_root_cid",
        "pgir_211_acceptance_cid", "pgir_211_verifier_run_cid", "portability_receipt_cid",
        "artifact_count", "artifacts", "immutability", "decision",
        "descendant_execution_authorized", "manifest_cid",
    }, "manifest field population drift")
    require(set(result) == {
        "schema", "task_id", "objective_revision", "repository_id", "source_tree_id",
        "result_identity", "campaign_input_root_cid", "manifest_cid", "revision_set_cid",
        "plan_admission_receipt_id", "verification_receipt_cid", "previous_root_cid",
        "pgir_211_acceptance_cid", "pgir_211_verifier_run_cid", "portability_receipt_cid",
        "disposition", "decision", "completion_authoritative", "descendant_execution_authorized",
        "training_task_eligible_count", "unresolved_identities", "reason_codes", "portability",
        "rollback", "result_sha256", "result_cid",
    }, "result field population drift")
    require(set(verification) == {
        "schema", "verifier_interface", "campaign_input_root_cid", "revision_set_cid",
        "plan_admission_receipt_id", "campaign_decision", "verification_verdict",
        "all_integrity_checks_passed", "authorizes_execution", "checks", "receipt_cid",
    }, "verification receipt field population drift")
    require(set(manifest["artifacts"]) == MANIFEST_ARTIFACTS, "manifest artifact population drift")
    require(manifest["artifact_count"] == 10, "manifest artifact count drift")
    expected_artifacts = {}
    for name in sorted(MANIFEST_ARTIFACTS):
        data = (FREEZE_DIR / name).read_bytes()
        expected_artifacts[name] = {"raw_cid": raw_cid(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    require(manifest["artifacts"] == expected_artifacts, "manifest artifact identities drift")
    links = {
        "campaign_input_root_cid": root["root_cid"],
        "revision_set_cid": revisions["revision_set_cid"],
        "plan_admission_receipt_id": admission["receipt_id"],
        "verification_receipt_cid": verification["receipt_cid"],
        "pgir_211_verifier_run_cid": run["run_cid"],
        "portability_receipt_cid": portability["receipt_cid"],
    }
    for field, value in links.items():
        require(manifest[field] == value, f"manifest {field} drift")
        require(result[field] == value, f"result {field} drift")
    require(root["gates"]["portability"]["receipt_cid"] == portability["receipt_cid"], "root portability gate link drift")
    require(root["integrated_evidence"]["fresh_verifier_run"] == {"run_cid": run["run_cid"], "mode": "network", "exit_code": 0, "stdout": run["stdout"]}, "root fresh verifier-run envelope drift")
    checks = {row["check_id"]: row for row in verification["checks"]}
    require(len(checks) == len(verification["checks"]) == 10, "verification check population drift")
    require(set(checks) == {
        "previous-root-cid", "canonical-root-identity", "pgir-211-result", "pgir-211-later-forest",
        "fresh-pgir-211-verifier-run", "rights-corpus-holdout-tokenizer-baseline-integrated-portability-gates",
        "portability-documented-no-go", "descendant-task-revisions", "supervisor-plan-admission", "fail-closed-lease-barrier",
    }, "verification check IDs drift")
    expected_checks = [
        {"check_id": "previous-root-cid", "status": "passed", "evidence": HISTORICAL_ROOT_CID},
        {"check_id": "canonical-root-identity", "status": "passed", "evidence": root["root_cid"]},
        {"check_id": "pgir-211-result", "status": "passed", "evidence": P211_CID},
        {"check_id": "pgir-211-later-forest", "status": "passed", "evidence": PGIR_211_COMPLETION},
        {"check_id": "fresh-pgir-211-verifier-run", "status": "passed", "evidence": run["run_cid"]},
        {"check_id": "rights-corpus-holdout-tokenizer-baseline-integrated-portability-gates", "status": "passed", "evidence": "all seven gates evaluated; none authorize learned execution"},
        {"check_id": "portability-documented-no-go", "status": "passed", "evidence": portability["receipt_cid"]},
        {"check_id": "descendant-task-revisions", "status": "passed", "evidence": revisions["revision_set_cid"]},
        {"check_id": "supervisor-plan-admission", "status": "passed", "evidence": admission["receipt_id"]},
        {"check_id": "fail-closed-lease-barrier", "status": "passed", "evidence": "no-go root revises 2 descendants with zero lease-eligible tasks"},
    ]
    require(verification["checks"] == expected_checks, "verification check order or evidence drift")
    require(checks["portability-documented-no-go"]["evidence"] == portability["receipt_cid"], "verification portability evidence drift")
    require(checks["fresh-pgir-211-verifier-run"]["evidence"] == run["run_cid"], "verification run evidence drift")
    require(checks["descendant-task-revisions"]["evidence"] == revisions["revision_set_cid"], "verification revision evidence drift")
    require(checks["supervisor-plan-admission"]["evidence"] == admission["receipt_id"], "verification admission evidence drift")
    require(all(row == {"check_id": row["check_id"], "status": "passed", "evidence": row["evidence"]} for row in verification["checks"]), "verification check envelope drift")
    require(verification["campaign_input_root_cid"] == root["root_cid"], "verification root link drift")
    require(verification["revision_set_cid"] == revisions["revision_set_cid"], "verification revision link drift")
    require(verification["plan_admission_receipt_id"] == admission["receipt_id"], "verification admission link drift")
    require(
        {key: verification[key] for key in (
            "schema", "verifier_interface", "campaign_decision", "verification_verdict",
            "all_integrity_checks_passed", "authorizes_execution",
        )}
        == {
            "schema": "PGIRFreezeVerificationReceipt@1",
            "verifier_interface": "pgir-successor-freeze-independent-verifier/v1",
            "campaign_decision": "no_go", "verification_verdict": "verified",
            "all_integrity_checks_passed": True, "authorizes_execution": False,
        },
        "verification receipt verdict envelope drift",
    )
    require(
        {key: manifest[key] for key in ("schema", "task_id", "previous_root_cid", "pgir_211_acceptance_cid", "immutability", "decision", "descendant_execution_authorized")}
        == {
            "schema": "PGIRSuccessorFreezeBundleManifest@1", "task_id": "PGIR-205",
            "previous_root_cid": HISTORICAL_ROOT_CID, "pgir_211_acceptance_cid": P211_CID,
            "immutability": "supersede_never_overwrite", "decision": "no_go",
            "descendant_execution_authorized": False,
        },
        "manifest disposition drift",
    )
    require(
        {key: result[key] for key in (
            "schema", "task_id", "objective_revision", "repository_id", "source_tree_id",
            "result_identity", "previous_root_cid", "pgir_211_acceptance_cid", "disposition",
            "decision", "completion_authoritative", "descendant_execution_authorized",
            "training_task_eligible_count", "unresolved_identities", "reason_codes", "portability", "rollback",
        )}
        == {
            "schema": "pgir-task-result@1", "task_id": "PGIR-205", "objective_revision": OBJECTIVE_REVISION,
            "repository_id": REPOSITORY_ID, "source_tree_id": PGIR_211_COMPLETION_TREE,
            "result_identity": "RESULT(PGIR-205)", "previous_root_cid": HISTORICAL_ROOT_CID,
            "pgir_211_acceptance_cid": P211_CID, "disposition": "frozen_no_go", "decision": "no_go",
            "completion_authoritative": False, "descendant_execution_authorized": False,
            "training_task_eligible_count": 0, "unresolved_identities": [],
            "reason_codes": root["qualification"]["reason_codes"], "portability": "portability_no_go",
            "rollback": "retain this immutable successor root and create a separately admitted superseding root",
        },
        "result disposition drift",
    )
    require(result["manifest_cid"] == manifest["manifest_cid"], "result manifest link drift")


def verify_admission(admission: Mapping[str, Any], root: Mapping[str, Any], revisions: Mapping[str, Any]) -> None:
    require(set(admission) == {
        "schema", "compiler_version", "requirement_id", "request_id", "candidate_plan_id",
        "candidate_graph_id", "repository_tree_id", "verdict", "admitted", "semantic_roots",
        "intent_result_id", "legal_result_ids", "legal_permission_ids", "security_decision_ids",
        "security_grant_ids", "checked_dependency_ids", "checked_assumption_ids",
        "generated_formula_ids", "proof_result_ids", "checked_validation_ids",
        "cve_security_evidence_ids", "rejection_reasons", "reason_codes", "counterexamples",
        "local_replan_action_ids", "closure_id", "permissions_are_grants",
        "generated_formulas_are_proofs", "authorizes_execution", "receipt_id",
    }, "plan-admission field population drift")
    require(admission["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-admission-receipt@1", "admission schema drift")
    require(admission["compiler_version"] == 1 and admission["requirement_id"] == IR_CONFORMANCE_REQUIREMENT_ID, "admission compiler contract drift")
    require(admission["candidate_plan_id"] == revisions["revision_set_cid"] and admission["candidate_graph_id"] == revisions["candidate_graph_cid"], "admission candidate links drift")
    require(admission["repository_tree_id"] == PGIR_211_COMPLETION_TREE, "admission repository tree drift")
    request_projection = {
        "campaign_input_root_cid": root["root_cid"], "candidate_plan_id": revisions["revision_set_cid"],
        "candidate_graph_id": revisions["candidate_graph_cid"], "repository_tree_id": PGIR_211_COMPLETION_TREE,
    }
    require(admission["request_id"] == supervisor_identity("pgir-campaign-plan-admission-request", request_projection), "admission request identity drift")
    require(admission["verdict"] == "rejected" and admission["admitted"] is False and admission["authorizes_execution"] is False, "admission fail-closed verdict drift")
    require(admission["permissions_are_grants"] is False and admission["generated_formulas_are_proofs"] is False, "admission authority booleans drift")
    require(admission["semantic_roots"] == {
        "campaign": root["root_cid"], "compiler": root["bindings"]["compiler"]["binding_cid"],
        "corpus": root["bindings"]["corpus"]["binding_cid"], "decompiler": root["bindings"]["decompiler"]["binding_cid"],
        "examples": root["bindings"]["example_contracts"]["binding_cid"], "policy": root["bindings"]["policy"]["binding_cid"],
        "schema": root["bindings"]["schema_registry"]["binding_cid"], "source": root["bindings"]["source_snapshots"]["binding_cid"],
        "split": root["bindings"]["split"]["binding_cid"], "tokenizer_policy": root["bindings"]["tokenizer_policy"]["binding_cid"],
    }, "admission semantic-root links drift")
    require(admission["checked_dependency_ids"] == ["dependency:PGIR-206", "dependency:PGIR-207"], "admission dependency population drift")
    require(admission["checked_assumption_ids"] == ["assumption:all-required-identities-resolved", "assumption:training-corpus-admitted"], "admission assumption population drift")
    require(admission["checked_validation_ids"] == [
        "validation:current-baseline", "validation:integrated-evidence", "validation:lineage-leakage",
        "validation:portability", "validation:referential-integrity", "validation:rights-admission", "validation:tokenizer-admission",
    ], "admission validation population drift")
    for empty in ("intent_result_id", "closure_id"):
        require(admission[empty] == "", f"admission {empty} drift")
    for empty in ("legal_result_ids", "legal_permission_ids", "security_decision_ids", "security_grant_ids", "generated_formula_ids", "proof_result_ids", "cve_security_evidence_ids", "counterexamples"):
        require(admission[empty] == [], f"admission {empty} population drift")
    reasons = admission["rejection_reasons"]
    require(len(reasons) == 2 and admission["reason_codes"] == ["assumption_unresolved", "validation_failed"], "typed rejection population drift")
    require(reasons == sorted(reasons, key=lambda row: row["rejection_id"]), "typed rejections are not canonically ordered")

    def rejection(*, code: str, domain: str, message: str, source_ids: Sequence[str], details: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema": "ipfs_accelerate_py/agent-supervisor/plan-admission-rejection@1",
            "code": code, "domain": domain, "message": message,
            "action_id": "", "effect_id": "", "dependency_id": "", "obligation_id": "",
            "source_ids": sorted(set(source_ids)), "details": dict(details),
        }
        return {**body, "rejection_id": supervisor_identity("plan-admission-rejection", body)}

    expected_reasons = sorted(
        [
            rejection(
                code="assumption_unresolved", domain="assumption",
                message="No source row has rights authority for learned training under this freeze.",
                source_ids=(root["root_cid"], root["bindings"]["rights"]["binding_cid"]),
                details={"training_admitted_rows": 0, "quarantined_source_count": 7173},
            ),
            rejection(
                code="validation_failed", domain="validation",
                message="Rights, corpus, holdout, tokenizer, current-baseline, integrated-evidence, and portability gates are not all passed.",
                source_ids=(root["root_cid"], root["bindings"]["split"]["binding_cid"], root["bindings"]["tokenizer_policy"]["binding_cid"]),
                details={
                    "failed_gates": sorted(root["gates"]),
                    "insufficient_holdouts": root["qualification"]["insufficient_holdouts"],
                    "portability_status": "portability_no_go",
                    "semantic_baseline_currently_qualified": False,
                },
            ),
        ],
        key=lambda row: row["rejection_id"],
    )
    require(reasons == expected_reasons, "typed rejection bodies drift")
    require(
        admission["local_replan_action_ids"]
        == [
            "replan:admit-rights-qualified-corpus", "replan:complete-required-holdouts",
            "replan:publish-or-document-portability-no-go", "replan:qualify-current-semantic-baseline",
            "replan:supersede-campaign-freeze-root",
        ],
        "admission replan population drift",
    )
    for rejection in reasons:
        require(set(rejection) == {"schema", "code", "domain", "message", "action_id", "effect_id", "dependency_id", "obligation_id", "source_ids", "details", "rejection_id"}, "typed rejection field population drift")
        require(rejection["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-admission-rejection@1", "typed rejection schema drift")
        projection = dict(rejection)
        claimed = projection.pop("rejection_id")
        require(claimed == supervisor_identity("plan-admission-rejection", projection), "typed rejection identity drift")
        require(rejection["source_ids"] == sorted(set(rejection["source_ids"])), "typed rejection source population drift")


def verify_root_semantics(
    root: Mapping[str, Any], historical: Mapping[str, Any], run: Mapping[str, Any]
) -> None:
    require(
        root["repository"]
        == {
            "repository_id": REPOSITORY_ID, "source_revision": PGIR_211_COMPLETION,
            "source_tree_id": PGIR_211_COMPLETION_TREE, "datasets_commit": NESTED_CURRENT,
            "datasets_tree_id": NESTED_CURRENT_TREE, "source_set_id": "SRCSET-1",
        },
        "root repository baseline drift",
    )
    require(
        root["canonicalization"]
        == {
            "identity_projection": "entire document excluding root_cid and root_sha256",
            "json": "UTF-8; sorted keys; compact separators; ensure_ascii=false; no floats",
            "cid": "CIDv1/base32/dag-json/sha2-256",
            "rendering": "two-space indentation and one terminal LF",
        },
        "root canonicalization contract drift",
    )
    require(
        root["referential_integrity"]
        == {
            "all_required_bindings_present": True,
            "required_binding_names": sorted(root["bindings"]),
            "unresolved_identities": [], "compiler_alias_resolved": True,
            "decompiler_alias_resolved": True, "hidden_labels_accessed": False,
            "source_or_split_mutated": False,
        },
        "root referential-integrity envelope drift",
    )
    require(set(root["bindings"]) == INHERITED_BINDING_NAMES | {"rights", "corpus", "lineage", "split", "tokenizer_policy"}, "root binding-name population drift")

    rights = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/rights_manifest.json")
    corpus = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json")
    lineage = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/lineage_graph.json")
    split = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json")
    holdouts = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/holdout_report.json")
    tokenizer = strict_json(FREEZE_DIR / "tokenizer/tokenizer_policy.json")
    tokenizer_result = strict_json(FREEZE_DIR / "tokenizer/result.json")

    def bound(payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "binding_cid": dag_json_cid(payload)}

    expected_new = {
        "rights": bound({
            "task_id": "PGIR-200", "result_identity": "RESULT(PGIR-200)",
            "admission_decision": rights["admission_decision"],
            "training_admitted_rows": rights["training_admitted_rows"],
            "quarantined_source_count": rights["quarantined_source_record_count"],
            "pinset_id": rights["pinset_id"], "file": root["bindings"]["rights"]["file"],
        }),
        "corpus": bound({
            "task_id": "PGIR-201", "result_identity": "RESULT(PGIR-201)",
            "manifest_id": corpus["manifest_id"], "materialized": corpus["materialized"],
            "source_count": 7173, "training_admitted_rows": corpus["counts"]["admitted_source_rows"],
            "materialized_source_rows": corpus["counts"]["materialized_source_rows"],
            "file": root["bindings"]["corpus"]["file"],
        }),
        "lineage": bound({
            "task_id": "PGIR-201", "result_identity": "RESULT(PGIR-201)",
            "graph_id": lineage["graph_id"], "admitted_lineage_groups": lineage["admitted_lineage_groups"],
            "materialized_row_count": lineage["validation"]["materialized_row_count"],
            "file": root["bindings"]["lineage"]["file"],
        }),
        "split": bound({
            "task_id": "PGIR-202", "result_identity": "RESULT(PGIR-202)", "status": split["status"],
            "leakage_passed": split["leakage_passed"], "hidden_test_commitment": split["hidden_test_commitment"],
            "holdouts": {name: {"status": value["status"], "count": value["count"]} for name, value in holdouts["holdouts"].items() if name in REQUIRED_HOLDOUTS},
            "file": root["bindings"]["split"]["file"], "holdout_report": root["bindings"]["split"]["holdout_report"],
        }),
        "tokenizer_policy": bound({
            "task_id": "PGIR-203", "result_identity": "RESULT(PGIR-203)",
            "policy_cid": tokenizer["policy_cid"], "result_cid": tokenizer_result["result_cid"],
            "status": tokenizer["status"], "unknown_token_behavior": tokenizer["unknown_token_behavior"],
            "training_authorized": tokenizer_result["training_authorized"],
            "file": root["bindings"]["tokenizer_policy"]["file"],
            "result_file": root["bindings"]["tokenizer_policy"]["result_file"],
        }),
    }
    require({name: root["bindings"][name] for name in expected_new} == expected_new, "new binding semantic projections drift")
    require(
        root["supersession"]
        == {
            "immutable": True, "previous_root_cid": HISTORICAL_ROOT_CID,
            "previous_root_schema": "IRCampaignInputRoot@1",
            "previous_root_path": "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json",
            "previous_root_file": root["supersession"]["previous_root_file"],
            "replacement_policy": "supersede_never_overwrite",
        },
        "supersession envelope drift",
    )
    require(root["supersession"]["previous_root_file"] == next(
        value for value in _collect_full_bindings(root) if value["path"].endswith("freeze/campaign_input_root.json")
    ), "previous-root file binding drift")
    integrated = strict_json(FREEZE_DIR / "integrated-acceptance/integrated_acceptance.json")
    require(
        root["integrated_evidence"]
        == {
            "result_identity": "RESULT(PGIR-211)", "acceptance_cid": P211_CID,
            "predecessor_acceptance_cids": {"PGIR-208": P208_CID, "PGIR-209": P209_CID, "PGIR-210": P210_CID},
            "supersession_chain": ["PGIR-208", "PGIR-210", "PGIR-211"],
            "decision": integrated["decision"], "completion_authoritative": integrated["completion_authoritative"],
            "pgir_205_execution_authorized": False, "file": root["integrated_evidence"]["file"],
            "later_forest": root["integrated_evidence"]["later_forest"],
            "fresh_verifier_run": {"run_cid": run["run_cid"], "mode": "network", "exit_code": 0, "stdout": run["stdout"]},
        },
        "integrated-evidence envelope drift",
    )


def _collect_full_bindings(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, dict):
        if "revision" in value and {"path", "repository", "raw_cid", "sha256", "size_bytes", "git_blob"} <= set(value):
            found.append(value)
        for child in value.values():
            found.extend(_collect_full_bindings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_full_bindings(child))
    return found


def verify_gates(
    root: Mapping[str, Any], run: Mapping[str, Any], portability: Mapping[str, Any]
) -> None:
    gates = root["gates"]
    require(set(gates) == set(GATE_NAMES), "gate population drift")
    for name in GATE_NAMES:
        require(gates[name]["passed"] is False, f"{name} gate unexpectedly passed")
    require(root["qualification"]["decision"] == "no_go", "qualification is not no-go")
    require(root["qualification"]["descendant_execution_authorized"] is False, "descendants were authorized")
    require(root["qualification"]["lease_barrier"] == "closed", "lease barrier is not closed")
    require(root["qualification"]["training_task_eligible_count"] == 0, "learned tasks were marked eligible")
    require(root["qualification"]["training_admitted_rows"] == 0, "training admitted rows drifted")
    require(root["qualification"]["rights_quarantined_rows"] == 7173, "quarantine count drifted")
    require(root["qualification"]["corpus_materialized"] is False, "corpus materialized")
    require(
        root["qualification"]["insufficient_holdouts"] == list(REQUIRED_HOLDOUTS),
        "insufficient holdout population drift",
    )
    require("portability_no_go" in root["qualification"]["reason_codes"], "portability no-go was not documented")
    require(root["integrated_evidence"]["pgir_205_execution_authorized"] is False, "integrated evidence authorized execution")
    require(root["integrated_evidence"]["acceptance_cid"] == P211_CID, "integrated evidence CID drift")
    require(
        root["integrated_evidence"]["predecessor_acceptance_cids"]
        == {"PGIR-208": P208_CID, "PGIR-209": P209_CID, "PGIR-210": P210_CID},
        "predecessor acceptance CID drift",
    )
    rights = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/rights_manifest.json")
    corpus = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json")
    split = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json")
    holdouts = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/holdout_report.json")
    tokenizer = strict_json(FREEZE_DIR / "tokenizer/tokenizer_policy.json")
    tokenizer_result = strict_json(FREEZE_DIR / "tokenizer/result.json")
    retirement = strict_json(REPOSITORY_ROOT / "ipfs_datasets_py/data/ir_learning/evaluations/deterministic/successor-v1/retirement_receipt.json")
    baseline = strict_json(FREEZE_DIR / "baseline-acceptance/baseline_acceptance.json")
    integrated = strict_json(FREEZE_DIR / "integrated-acceptance/integrated_acceptance.json")
    failed_holdouts = [name for name in REQUIRED_HOLDOUTS if (holdouts["holdouts"].get(name) or {}).get("status") != "populated"]
    expected = {
        "rights": {
            "passed": False, "result_identity": "RESULT(PGIR-200)",
            "training_admitted_rows": rights["training_admitted_rows"],
            "training_eligible": rights["training_eligible"],
            "admission_decision": rights["admission_decision"],
            "file": root["bindings"]["rights"]["file"],
        },
        "corpus": {
            "passed": False, "result_identity": "RESULT(PGIR-201)",
            "materialized": corpus["materialized"],
            "admitted_source_rows": corpus["counts"]["admitted_source_rows"],
            "materialized_source_rows": corpus["counts"]["materialized_source_rows"],
            "file": root["bindings"]["corpus"]["file"],
        },
        "holdouts": {
            "passed": False, "result_identity": "RESULT(PGIR-202)",
            "required_holdouts": list(REQUIRED_HOLDOUTS), "failed_holdouts": failed_holdouts,
            "leakage_passed": split["leakage_passed"],
            "hidden_test_commitment": split["hidden_test_commitment"], "hidden_labels_accessed": False,
            "file": root["bindings"]["split"]["file"],
            "holdout_report": root["bindings"]["split"]["holdout_report"],
        },
        "tokenizer": {
            "passed": False, "result_identity": "RESULT(PGIR-203)",
            "result_cid": tokenizer_result["result_cid"], "status": tokenizer["status"],
            "policy_cid": tokenizer["policy_cid"], "training_authorized": tokenizer_result["training_authorized"],
            "file": root["bindings"]["tokenizer_policy"]["file"],
            "result_file": root["bindings"]["tokenizer_policy"]["result_file"],
        },
        "current_baseline": {
            "passed": False, "result_identity": "RESULT(PGIR-204)", "adjudication_identity": "RESULT(PGIR-209)",
            "retirement_cid": retirement["retirement_cid"], "decision": retirement["decision"]["status"],
            "current_input_qualified_r1_cid": retirement["acceptance"]["current_input_qualified_r1_cid"],
            "retirement_file": root["gates"]["current_baseline"]["retirement_file"],
            "baseline_file": root["gates"]["current_baseline"]["baseline_file"],
        },
        "integrated_evidence": {
            "passed": False, "result_identity": "RESULT(PGIR-211)", "acceptance_cid": integrated["acceptance_cid"],
            "decision": integrated["decision"], "completion_authoritative": integrated["completion_authoritative"],
            "pgir_205_execution_authorized": integrated["pgir_205_execution_authorized"],
            "fresh_verifier_run_cid": run["run_cid"], "fresh_verifier_mode": run["mode"],
            "file": root["integrated_evidence"]["file"],
        },
        "portability": {
            "passed": False, "receipt_cid": portability["receipt_cid"], "status": portability["status"],
            "missing_outer_commits": portability["missing_outer_commits"],
            "missing_nested_commits": portability["missing_nested_commits"],
            "documented_no_go_required": True,
        },
    }
    require(gates == expected, "seven-gate recomputation drift")
    reason_codes = [
        "no_rights_admitted_training_rows", "corpus_not_materialized",
        "required_holdouts_insufficient", "tokenizer_not_admitted",
        "historical_semantic_baseline_not_currently_qualified",
        "integrated_evidence_does_not_authorize_execution", "portability_no_go",
    ]
    require(
        root["qualification"]
        == {
            "decision": "no_go", "lease_barrier": "closed", "descendant_execution_authorized": False,
            "training_task_eligible_count": 0, "training_admitted_rows": 0,
            "rights_quarantined_rows": 7173, "corpus_materialized": False,
            "leakage_passed": True, "insufficient_holdouts": list(REQUIRED_HOLDOUTS),
            "reason_codes": reason_codes,
        },
        "qualification recomputation drift",
    )


def verify(*, fresh_network: bool = False) -> None:
    artifacts = {name: strict_json(FREEZE_DIR / name) for name in REQUIRED_JSON}
    root = artifacts["campaign_input_root.json"]
    revisions = artifacts["descendant_task_revisions.json"]
    admission = artifacts["plan_admission_receipt.json"]
    run = artifacts["pgir_211_baseline_verifier_run.json"]
    portability = artifacts["portability_receipt.json"]
    verification = artifacts["verification_receipt.json"]
    manifest = artifacts["manifest.json"]
    result = artifacts["result.json"]

    verify_bundle_population(artifacts)
    verify_root_schema(root)

    require(root["schema"] == "IRCampaignInputRoot@2", "root schema drift")
    require(root["contract_version"] == 2, "root contract version drift")
    require(root["task_id"] == "PGIR-205", "root task drift")
    require(root["objective_id"] == "PGIR-205" and root["objective_revision"] == OBJECTIVE_REVISION, "root objective identity drift")
    require(
        root["interface"] == "proof-grounded-ir-learning/campaign-input-root/v2",
        "root interface drift",
    )
    verify_projection_identity(root, cid_field="root_cid", sha_field="root_sha256")
    verify_projection_identity(revisions, cid_field="revision_set_cid", sha_field="revision_set_sha256")
    verify_projection_identity(run, cid_field="run_cid", sha_field="run_sha256")
    verify_projection_identity(portability, cid_field="receipt_cid", sha_field="receipt_sha256")
    verify_projection_identity(verification, cid_field="receipt_cid")
    verify_projection_identity(manifest, cid_field="manifest_cid")
    verify_projection_identity(result, cid_field="result_cid", sha_field="result_sha256")

    historical = strict_json(HISTORICAL_ROOT)
    require(historical["root_cid"] == HISTORICAL_ROOT_CID, "historical freeze CID drift")
    require(historical["root_sha256"] == HISTORICAL_ROOT_SHA256, "historical freeze SHA drift")
    require(root["supersession"]["previous_root_cid"] == HISTORICAL_ROOT_CID, "previous_root_cid does not bind the historical freeze")
    require(root["supersession"]["previous_root_cid"] == historical["root_cid"], "previous_root_cid does not equal the live historical root")
    require(root["supersession"]["immutable"] is True, "supersession is not immutable")
    require(root["supersession"]["replacement_policy"] == "supersede_never_overwrite", "replacement policy drift")
    projection = {key: value for key, value in historical.items() if key not in {"root_cid", "root_sha256"}}
    require(historical["root_cid"] == dag_json_cid(projection), "historical freeze CID does not replay")
    require(
        historical["root_sha256"] == "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest(),
        "historical freeze SHA does not replay",
    )
    require(
        {name: root["bindings"][name] for name in INHERITED_BINDING_NAMES}
        == {name: historical["bindings"][name] for name in INHERITED_BINDING_NAMES},
        "inherited historical binding population or bytes drift",
    )
    verify_recursive_file_bindings(root)
    verify_legacy_file_bindings({name: root["bindings"][name] for name in INHERITED_BINDING_NAMES})
    verify_root_semantics(root, historical, run)

    require(root["referential_integrity"]["hidden_labels_accessed"] is False, "hidden labels were accessed")
    require(root["referential_integrity"]["source_or_split_mutated"] is False, "source or split mutated")
    require(
        set(root["referential_integrity"]["required_binding_names"]) == set(root["bindings"]),
        "binding name population drift",
    )
    require(len(root["bindings"]) == 12, "required binding count drift")
    for name, binding in root["bindings"].items():
        require("binding_cid" in binding, f"{name} missing binding_cid")
        verify_projection_identity(binding, cid_field="binding_cid")

    rights = root["bindings"]["rights"]
    corpus = root["bindings"]["corpus"]
    split = root["bindings"]["split"]
    tokenizer = root["bindings"]["tokenizer_policy"]
    require(rights["training_admitted_rows"] == 0, "rights admitted rows drifted")
    require(rights["quarantined_source_count"] == 7173, "rights quarantine drifted")
    require(corpus["materialized"] is False, "corpus materialized")
    require(corpus["training_admitted_rows"] == 0, "corpus admitted rows drifted")
    require(split["leakage_passed"] is True, "leakage failed")
    require(split["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT, "hidden-test commitment drift")
    require(tokenizer["training_authorized"] is False, "tokenizer authorized training")
    require(tokenizer["unknown_token_behavior"] == "fail_closed", "unknown token behavior drift")
    verify_file_binding(rights["file"])
    verify_file_binding(corpus["file"])
    verify_file_binding(root["bindings"]["lineage"]["file"])
    verify_file_binding(split["file"])
    verify_file_binding(split["holdout_report"])
    verify_file_binding(tokenizer["file"])
    verify_file_binding(root["integrated_evidence"]["file"])
    verify_file_binding(root["gates"]["current_baseline"]["retirement_file"])
    verify_file_binding(root["gates"]["current_baseline"]["baseline_file"])

    live_rights = strict_json(
        DATASETS_ROOT / "data/ir_learning/corpora/successor-v1/rights_manifest.json"
    )
    live_corpus = strict_json(
        DATASETS_ROOT / "data/ir_learning/corpora/successor-v1/corpus_root.json"
    )
    live_split = strict_json(
        DATASETS_ROOT / "data/ir_learning/splits/successor-v1/split_root.json"
    )
    live_holdouts = strict_json(
        DATASETS_ROOT / "data/ir_learning/splits/successor-v1/holdout_report.json"
    )
    live_tokenizer = strict_json(FREEZE_DIR / "tokenizer/tokenizer_policy.json")
    live_integrated = strict_json(FREEZE_DIR / "integrated-acceptance/integrated_acceptance.json")
    require(live_rights["training_admitted_rows"] == 0, "live rights admitted rows drifted")
    require(live_corpus["materialized"] is False, "live corpus materialized")
    require(live_split["status"] == "permanent_no_go", "live split is not permanent no-go")
    require(live_holdouts["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT, "live hidden-test commitment drift")
    require(
        all((live_holdouts["holdouts"][name]["status"] == "permanent_no_go") for name in REQUIRED_HOLDOUTS),
        "a required holdout is no longer permanent_no_go",
    )
    require(live_tokenizer["status"] == "permanently_deterministic_only", "live tokenizer status drift")
    require(live_integrated["acceptance_cid"] == P211_CID, "live PGIR-211 CID drift")
    require(live_integrated["pgir_205_execution_authorized"] is False, "live PGIR-211 authorized PGIR-205")

    verify_forest(root["integrated_evidence"]["later_forest"])
    verify_gates(root, run, portability)
    verify_portability(portability)
    verify_pgir_211_run(run, fresh_network=fresh_network)

    verify_descendant_revisions(revisions, root)
    verify_admission(admission, root, revisions)

    require(revisions["lease_eligible_count"] == 0, "descendant leases were opened")
    require(revisions["descendant_task_count"] == 2, "descendant count drift")
    require(
        [item["task_id"] for item in revisions["revisions"]] == ["PGIR-206", "PGIR-207"],
        "descendant identity drift",
    )
    require(all(item["lease_eligible"] is False for item in revisions["revisions"]), "a descendant is lease-eligible")
    require(admission["verdict"] == "rejected", "plan admission was not rejected")
    require(admission["admitted"] is False, "plan admission admitted the freeze")
    require(admission["authorizes_execution"] is False, "plan admission authorized execution")
    admission_projection = dict(admission)
    claimed_receipt = admission_projection.pop("receipt_id")
    require(
        claimed_receipt == supervisor_identity("plan-admission-receipt", admission_projection),
        "plan admission receipt_id drift",
    )
    require(verification["authorizes_execution"] is False, "verification receipt authorized execution")
    require(verification["campaign_decision"] == "no_go", "verification decision drift")
    require(result["decision"] == "no_go", "result decision drift")
    require(result["descendant_execution_authorized"] is False, "result authorized descendants")
    require(result["completion_authoritative"] is False, "result claimed authoritative completion")
    require(result["previous_root_cid"] == HISTORICAL_ROOT_CID, "result previous_root_cid drift")
    require(result["pgir_211_acceptance_cid"] == P211_CID, "result PGIR-211 CID drift")
    require(result["training_task_eligible_count"] == 0, "result eligible-task count drift")
    require(result["portability"] == "portability_no_go", "result did not document portability_no_go")
    require(manifest["descendant_execution_authorized"] is False, "manifest authorized execution")
    require(manifest["previous_root_cid"] == HISTORICAL_ROOT_CID, "manifest previous_root_cid drift")
    require(manifest["pgir_211_acceptance_cid"] == P211_CID, "manifest PGIR-211 CID drift")
    require(
        manifest["portability_receipt_cid"] == portability["receipt_cid"],
        "manifest portability receipt link drift",
    )
    require(
        result["portability_receipt_cid"] == portability["receipt_cid"],
        "result portability receipt link drift",
    )

    verify_manifest_and_crosslinks(
        artifacts, root, revisions, admission, run, portability, verification, manifest, result
    )

    for name, binding in manifest["artifacts"].items():
        data = (FREEZE_DIR / name).read_bytes()
        require(binding["size_bytes"] == len(data), f"manifest size drift: {name}")
        require(
            binding["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(),
            f"manifest sha256 drift: {name}",
        )
        require(binding["raw_cid"] == raw_cid(data), f"manifest raw CID drift: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh-network",
        action="store_true",
        help="rerun the pinned PGIR-211 verifier in strict --network mode",
    )
    arguments = parser.parse_args(argv)
    try:
        verify(fresh_network=arguments.fresh_network)
    except FreezeVerificationError as exc:
        print(f"PGIR-205 successor freeze verification failed: {exc}", file=sys.stderr)
        return 1
    print("PGIR-205 successor freeze verifies; execution remains unauthorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
