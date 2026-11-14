"""Measure one recorded shared-owner Lake command without running it again.

The receipt and nonempty oleans live under the datasets tree. This module
reads those bytes, checks the pinned Lean 4.34.1 command, and refuses a
missing or changed artifact. The native owner is not the process that ran.
The preparation is the authored official-shape O/P fixture. A match is not
federal-corpus acceptance or legal acceptance.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .dependency_block_resolution import EXPECTED_SHARED_LAKE_BUILD_PROBE

_SHARED = "ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py"
_NATIVE = "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py"
_BUILD = "_lir1401_build"
_RECEIPT = _BUILD + "/build-receipt.json"
_INPUTS = _BUILD + "/project/definition-inputs.json"
_TOOLCHAIN_FILE = _BUILD + "/project/lean-toolchain"


def measure_shared_lake_build_probe() -> dict[str, object] | None:
    """Return the recorded fixture build, or None when the receipt drifts."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_SHARED_LAKE_BUILD_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    root = _datasets_root()
    receipt_path = root / _RECEIPT
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHARED_LAKE_BUILD_PROBE["receipt_sha256"]:
        raise ValueError("shared build receipt bytes changed")
    receipt = json.loads(raw)
    expected = EXPECTED_SHARED_LAKE_BUILD_PROBE
    if receipt.get("schema") != expected["schema"]:
        raise ValueError("shared build schema is not the shared owner schema")
    if receipt.get("schema") == "legal-corpus-v2-definition-use-native-build/v1":
        raise ValueError("native owner schema is not this command")
    if receipt.get("profile") != expected["profile"]:
        raise ValueError("shared build profile changed")
    if receipt.get("toolchain") != expected["toolchain"]:
        raise ValueError("shared build toolchain changed")
    if receipt.get("command") != expected["command"]:
        raise ValueError("shared build command changed")
    if receipt.get("returncode") != 0 or receipt.get("build_passed") is not True:
        raise ValueError("shared build did not pass")
    if receipt.get("lean_compilation_verified") is not True:
        raise ValueError("shared owner did not verify compilation")
    if receipt.get("backend_executed") is not True:
        raise ValueError("shared owner did not record a process")
    if (
        receipt.get("admitted") is not False
        or receipt.get("formalized") is not False
        or receipt.get("proof_authority") is not False
        or receipt.get("source_semantics_verified") is not False
        or receipt.get("independently_reviewed") is not False
        or receipt.get("training_qualified") is not False
        or receipt.get("context_closure_verified") is not False
    ):
        raise ValueError("shared build receipt claims authority")
    if (
        receipt.get("manifest_coverage_passed") is not True
        or receipt.get("binaries_unchanged") is not True
        or receipt.get("workspace_cleaned") is not True
        or receipt.get("timed_out") is not False
        or receipt.get("unavailable") is not False
        or receipt.get("output_truncated") is not False
        or receipt.get("execution_error") != ""
    ):
        raise ValueError("shared build closeout is incomplete")
    if receipt.get("compiled_modules") != receipt.get("expected_modules"):
        raise ValueError("compiled modules differ from the expected import closure")
    if receipt.get("expected_modules") != expected["expected_modules"]:
        raise ValueError("expected modules changed")
    if receipt.get("preparation_cid") != expected["preparation_cid"]:
        raise ValueError("preparation changed")
    if receipt.get("preparation_manifest_sha256") != expected["preparation_manifest_sha256"]:
        raise ValueError("preparation manifest changed")
    if receipt.get("binary_sha256") != expected["binary_sha256"]:
        raise ValueError("recorded binary hashes changed")
    axioms = receipt.get("theorem_axiom_dependencies")
    if type(axioms) is not dict or sorted(axioms) != list(expected["norm_theorems_without_axioms"]):
        raise ValueError("norm theorem set changed")
    if any(value != [] for value in axioms.values()):
        raise ValueError("a norm theorem depends on an axiom")
    combined = str(receipt.get("stdout") or "") + str(receipt.get("stderr") or "")
    if "sorry" in combined or "sorryAx" in combined or " admit" in combined:
        raise ValueError("sorry or admit text is present")
    _check_artifacts(root, receipt)
    _check_fixture(root)
    _check_binaries(expected["binary_sha256"])
    _check_owner_functions(root)
    toolchain = (root / _TOOLCHAIN_FILE).read_bytes()
    if toolchain != b"leanprover/lean4:v4.34.1\n":
        raise ValueError("project toolchain file changed")
    return dict(expected)


def _check_artifacts(root: Path, receipt: dict) -> None:
    records = EXPECTED_SHARED_LAKE_BUILD_PROBE["artifact_records"]
    files = receipt.get("artifact_files")
    hashes = receipt.get("artifact_sha256")
    if type(files) is not dict or type(hashes) is not dict or len(records) != 5:
        raise ValueError("artifact inventory changed")
    if set(files) != set(hashes) or len(files) != 5:
        raise ValueError("artifact inventory is incomplete")
    seen = set()
    for record in records:
        logical = ".lake/build/lib/lean/" + record["compiled_path"].removeprefix("compiled/")
        if files.get(logical) != record["compiled_path"]:
            raise ValueError("artifact path changed")
        if hashes.get(logical) != record["sha256"]:
            raise ValueError("receipt artifact hash changed")
        data = (root / _BUILD / record["compiled_path"]).read_bytes()
        if not data.startswith(b"olean") or len(data) != record["bytes"]:
            raise ValueError("olean bytes changed")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("olean hash changed")
        seen.add(record["module"])
    if seen != set(EXPECTED_SHARED_LAKE_BUILD_PROBE["expected_modules"]):
        raise ValueError("olean modules do not cover the import closure")


def _check_fixture(root: Path) -> None:
    text = (root / _INPUTS).read_text(encoding="utf-8")
    if '"kind":"authored_official_shape_fixture"' not in text:
        raise ValueError("fixture kind is not the authored official-shape fixture")
    for norm_id in EXPECTED_SHARED_LAKE_BUILD_PROBE["norm_ids"]:
        if '"norm_id":"' + norm_id + '"' not in text:
            raise ValueError("authored norm id is missing")
    if '"authenticity_verified":false' not in text or '"training_admitted":false' not in text:
        raise ValueError("fixture provenance flags changed")


def _check_binaries(expected: object) -> None:
    if type(expected) is not dict:
        raise ValueError("binary pin changed")
    for path, digest in expected.items():
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("pinned lake or lean binary changed")


def _check_owner_functions(root: Path) -> None:
    shared = ast.parse((root / _SHARED).read_text(encoding="utf-8"))
    native = ast.parse((root / _NATIVE).read_text(encoding="utf-8"))
    build = _function(shared, "build_shared_definition_corpus")
    rendered = ast.unparse(build)
    if "sorry" not in rendered or "sorryAx" not in rendered or "workspace_cleaned" not in rendered:
        raise ValueError("shared owner no longer checks closeout")
    if "'--offline'" not in rendered or "'--no-cache'" not in rendered or "'legal'" not in rendered:
        raise ValueError("shared owner no longer binds the legal command")
    if _function(native, "build_definition_use_native_v2") is None:
        raise ValueError("native owner entry is missing")


def _datasets_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        for root in (parent / "ipfs_datasets", parent / "external" / "ipfs_datasets"):
            if (root / _SHARED).is_file() and (root / _NATIVE).is_file() and (root / _RECEIPT).is_file():
                return root
    raise ValueError("shared lake build receipt is not on disk")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError("missing function " + name)
