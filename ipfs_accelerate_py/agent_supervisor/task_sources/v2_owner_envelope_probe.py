"""Measure two V2 definition-use owners without running Lake.

The native owner and the bounded build owner share one historical schema
label and keep different profiles. Each binds its command, module
population, hypotheses, and source files before its runner call. This
module only reads those sources.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .dependency_block_resolution import EXPECTED_V2_OWNER_ENVELOPE_PROBE

_SCHEMA = "legal-corpus-v2-definition-use-native-build/v1"
_NATIVE_PROFILE = "replayed-v2-definition-use-conditional-native-build/v1"
_BUILD_PROFILE = "bounded-v2-definition-use-conditional-build/v1"
_ARGV = ("--offline", "--no-cache", "build", "legal")


def measure_v2_owner_envelope_probe() -> dict[str, object] | None:
    """Return the fixed V2 owner envelope, or None when it drifts."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_V2_OWNER_ENVELOPE_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    native_path, build_path = _paths()
    native = _parse(native_path)
    build = _parse(build_path)
    native_fn = _function(native, "_build_validated_native")
    build_fn = _function(build, "build_definition_use_corpus_v2")
    receipt_fn = _function(build, "_receipt")
    native_run = _runner_call(native_fn)
    build_run = _runner_call(build_fn)
    native_command = _command_assignment(native_fn)
    build_command = _command_assignment(build_fn)
    if native_command is None or build_command is None:
        raise ValueError("exact offline legal command is not assigned")
    if not (native_command < native_run.lineno and build_command < build_run.lineno):
        raise ValueError("command is assigned after the runner call")
    if not _constant_before(native_fn, "expected_modules", native_run.lineno):
        raise ValueError("native module population is not bound before execution")
    if not _constant_before(native_fn, ".olean", native_run.lineno):
        raise ValueError("native artifact paths are not bound before execution")
    if not _name_before(native_fn, "source_files", native_run.lineno):
        raise ValueError("native source files are not bound before execution")
    if not _name_before(native_fn, "manifest", native_run.lineno):
        raise ValueError("native manifest is not snapshotted before execution")
    if not _subscript_name(native_fn, "manifest", "theorem_assumptions"):
        raise ValueError("native hypotheses are not taken from the manifest")
    if not _call_before(native_fn, "_installed_toolchain", native_run.lineno):
        raise ValueError("native toolchain pin is not checked before execution")
    if not (
        _call_before(build_fn, "_binary_expectations", build_run.lineno)
        and _call_before(build_fn, "_read_binary_pins", build_run.lineno)
    ):
        raise ValueError("build binary pins are not checked before execution")
    if not _call_before(build_fn, "validate_definition_use_corpus_v2", build_run.lineno):
        raise ValueError("build preparation is not validated before execution")
    if not _constant_before(build_fn, "expected_modules", build_run.lineno):
        raise ValueError("build module population is not bound before execution")
    if not _constant_before(build_fn, "expected_olean_paths", build_run.lineno):
        raise ValueError("build artifact population is not bound before execution")
    if not _runner_binds_checked_files(build_run):
        raise ValueError("build source files are not bound on the runner request")
    if not _manifest_field(receipt_fn, "theorem_assumptions"):
        raise ValueError("build hypotheses are not taken from the manifest")
    if not _call_name(build_fn, "_receipt"):
        raise ValueError("build receipt is not produced from the validated preparation")
    schema = _assigned_string(native, "BUILD_SCHEMA")
    build_schema = _assigned_string(build, "SCHEMA")
    native_profile = _assigned_string(native, "PROFILE")
    build_profile = _assigned_string(build, "PROFILE")
    if schema != build_schema or schema != _SCHEMA:
        raise ValueError("historical schema labels differ or drifted")
    if native_profile != _NATIVE_PROFILE or build_profile != _BUILD_PROFILE:
        raise ValueError("owner profiles drifted")
    if native_profile == build_profile:
        raise ValueError("owner profiles are interchangeable")
    if _assigned_int(native, "MAX_MODULES") != 6 or _assigned_int(build, "MAX_MODULES") != 6:
        raise ValueError("bounded module population drifted")
    if not _false_flag(native, "context_closure_verified"):
        raise ValueError("native context closure flag is not false")
    if not _dict_false(build, "toolchain_library_closure_verified"):
        raise ValueError("build library closure is not explicitly unverified")
    if not _dict_false(build, "admission_build_eligible"):
        raise ValueError("build admission eligibility is not false")
    native_doc = ast.get_docstring(native) or ""
    build_doc = ast.get_docstring(build) or ""
    if "Public import uses only the standard library." not in native_doc:
        raise ValueError("native import boundary drifted")
    if "Import performs no toolchain discovery or process execution." not in build_doc:
        raise ValueError("build import boundary drifted")
    if _module_level_runner(native) or _module_level_runner(build):
        raise ValueError("import invokes a runner")
    return {
        "admission_build_eligible": False,
        "admitted": False,
        "binary_sha256_recorded": False,
        "build_max_modules": 6,
        "build_profile": _BUILD_PROFILE,
        "command_argv_tail": list(_ARGV),
        "command_bound_before_execution": True,
        "context_closure_verified": False,
        "expected_artifact_population_bound_before_execution": True,
        "formalized": False,
        "hypotheses_bound_before_execution": True,
        "independent_binary_pin_required_before_execution": True,
        "lake_build_executed": False,
        "library_target": "legal",
        "native_max_modules": 6,
        "native_profile": _NATIVE_PROFILE,
        "owners_interchangeable": False,
        "profile_selected": False,
        "profiles_distinct": True,
        "proof_authority": False,
        "shared_schema_label": _SCHEMA,
        "source_closure_bound_before_execution": True,
        "source_semantics_verified": False,
        "toolchain_library_closure_verified": False,
    }


def _paths() -> tuple[Path, Path]:
    import ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_native_v2 as native

    root = Path(native.__file__).resolve().parent
    native_path = root / "legal_corpus_definition_use_native_v2.py"
    build_path = root / "legal_corpus_definition_use_build_v2.py"
    if not native_path.is_file() or not build_path.is_file():
        raise ValueError("both V2 owners are required")
    return native_path, build_path


def _parse(path: Path) -> ast.Module:
    raw = path.read_bytes()
    if not raw or len(raw) > 2_000_000:
        raise ValueError("owner file is empty or oversized")
    return ast.parse(raw.decode("utf-8"), filename=str(path))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError("missing function " + name)


def _assigned_string(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError("missing string " + name)


def _assigned_int(tree: ast.Module, name: str) -> int:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    raise ValueError("missing integer " + name)


def _is_legal_command(node: ast.AST) -> bool:
    if not isinstance(node, ast.Tuple) or len(node.elts) != 5:
        return False
    tails: list[str] = []
    for elt in node.elts[1:]:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
            return False
        tails.append(elt.value)
    return tuple(tails) == _ARGV


def _command_assignment(function: ast.FunctionDef) -> int | None:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "command"
            and _is_legal_command(node.value)
        ):
            return node.lineno
    return None


def _runner_call(function: ast.FunctionDef) -> ast.Call:
    found: list[ast.Call] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "run" or not isinstance(node.func.value, ast.Call):
            continue
        inner = node.func.value.func
        if isinstance(inner, ast.Attribute) and inner.attr == "BoundedToolRunner":
            found.append(node)
    if len(found) != 1:
        raise ValueError("one bounded runner call is required")
    return found[0]


def _constant_before(function: ast.FunctionDef, text: str, limit: int) -> bool:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Constant)
            and node.value == text
            and getattr(node, "lineno", limit) < limit
        ):
            return True
    return False


def _name_before(function: ast.FunctionDef, name: str, limit: int) -> bool:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
            and getattr(node, "lineno", limit) < limit
        ):
            return True
    return False


def _call_before(function: ast.FunctionDef, name: str, limit: int) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or getattr(node, "lineno", limit) >= limit:
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


def _call_name(function: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            return True
    return False


def _manifest_field(function: ast.FunctionDef, key: str) -> bool:
    if _subscript_name(function, "manifest", key):
        return True
    for node in ast.walk(function):
        if not isinstance(node, (ast.DictComp, ast.GeneratorExp)):
            continue
        indexes_manifest = False
        names_key = False
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == "manifest"
                and isinstance(child.slice, ast.Name)
            ):
                indexes_manifest = True
            if isinstance(child, ast.Constant) and child.value == key:
                names_key = True
        if indexes_manifest and names_key:
            return True
    return False


def _subscript_name(function: ast.FunctionDef, source: str, key: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        slc = node.slice
        if (
            isinstance(value, ast.Name)
            and value.id == source
            and isinstance(slc, ast.Constant)
            and slc.value == key
        ):
            return True
    return False


def _runner_binds_checked_files(call: ast.Call) -> bool:
    for node in ast.walk(call):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "files"
            and isinstance(node.value, ast.Name)
            and node.value.id == "checked"
        ):
            return True
    return False


def _false_flag(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "FALSE_FLAGS":
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            continue
        if value.func.attr != "fromkeys" or len(value.args) != 2:
            continue
        keys, flag = value.args
        if not isinstance(flag, ast.Constant) or flag.value is not False:
            return False
        return any(
            isinstance(elt, ast.Constant) and elt.value == name
            for elt in ast.walk(keys)
            if isinstance(elt, ast.Constant)
        )
    return False


def _dict_false(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == name
                and isinstance(value, ast.Constant)
                and value.value is False
            ):
                return True
    return False


def _module_level_runner(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            if child.func.attr == "BoundedToolRunner":
                return True
    return False
