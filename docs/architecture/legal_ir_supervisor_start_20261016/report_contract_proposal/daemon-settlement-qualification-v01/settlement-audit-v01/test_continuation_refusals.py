"""Authored cold/source controls; not a genuine original-daemon qualification.

Synthetic objects exercise refusal and diagnostic branches only. No positive
source closure, fake native owner, connection, daemon or callback is admitted.
Execute only inside a reviewed finite genuine native qualification envelope.
"""
import ast
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import threading
from types import ModuleType

import pytest


SOURCE = (Path(__file__).parent / "proposed/ipfs_accelerate_py/agent_supervisor/"
          "todo_daemon/original_finalizer_continuations.py")
SPEC = importlib.util.spec_from_file_location("cold_original_finalizer_source", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def refusal_registry():
    registry = object.__new__(MODULE.OriginalFinalizerContinuations)
    registry.pid, registry.thread = os.getpid(), threading.get_ident()
    registry.specifications = {stage: (None, None) for stage, _, _ in MODULE.STAGES}
    registry.entries = {}
    registry.rejected = None
    registry.pending = None
    registry.highest_registered = -1
    registry.draining = False
    return registry


def test_projected_module_is_refused():
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        MODULE.OriginalFinalizerContinuations({"name": MODULE.MODULE}, object())


def test_uninstalled_module_is_refused():
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        MODULE.OriginalFinalizerContinuations(ModuleType(MODULE.MODULE), object())


def test_bad_stage_retains_original_carrier_without_effect():
    registry = refusal_registry()
    callback = lambda: pytest.fail("refused callback must never run")
    original = {"actual_reference": object()}
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        registry.register("portable_unknown_stage", callback, None, original)
    carrier, _cause = registry.rejected
    assert carrier[1] is callback and carrier[3] is original
    assert registry.entries == {} and not registry.complete()


def test_dictionary_is_not_an_original_frame():
    registry = refusal_registry()
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        registry.register(MODULE.STAGES[0][0], lambda: None,
                          {"f_code": "original"}, {})
    assert registry.entries == {} and registry.rejected is not None


def test_live_foreign_frame_is_refused():
    registry = refusal_registry()
    frame = inspect.currentframe()
    try:
        with pytest.raises(MODULE.OriginalFinalizerRefusal):
            registry.register(MODULE.STAGES[0][0], lambda: None, frame, {})
        assert registry.entries == {}
    finally:
        del frame


def test_non_dictionary_reference_carrier_is_refused_and_retained():
    registry = refusal_registry()
    carrier = object()
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        registry.register(MODULE.STAGES[0][0], lambda: None, None, carrier)
    assert registry.rejected[0][3] is carrier


def test_duplicate_registration_is_refused_before_callback():
    registry = refusal_registry()
    stage = MODULE.STAGES[0][0]
    registry.entries[stage] = {"status": "done"}
    with pytest.raises(MODULE.OriginalFinalizerRefusal,
                       match="stage_already_registered"):
        registry.register(stage, lambda: pytest.fail("must not run"), None, {})
    assert registry.entries[stage]["status"] == "done"


def test_inner_registration_after_outer_is_refused():
    registry = refusal_registry()
    registry.highest_registered = 1
    with pytest.raises(MODULE.OriginalFinalizerRefusal,
                       match="registration_order_changed"):
        registry.register(MODULE.STAGES[0][0], lambda: None, None, {})


def test_first_rejected_carrier_is_preserved_on_further_refusal():
    registry = refusal_registry()
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        registry.register("first", None, None, {})
    first = registry.rejected
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        registry.register("second", None, None, {})
    assert registry.rejected is first


def test_rejected_registry_cannot_drain():
    registry = refusal_registry()
    with pytest.raises(MODULE.OriginalFinalizerRefusal):
        registry.register("bad", None, None, {})
    with pytest.raises(MODULE.OriginalFinalizerPending) as caught:
        registry.drain()
    assert caught.value.actual_registry is registry
    assert registry.entries == {} and not registry.complete()


def test_existing_pending_is_not_replayed_or_replaced():
    registry = refusal_registry()
    cause = RuntimeError("original failure")
    pending = MODULE.OriginalFinalizerPending(registry, cause)
    registry.pending = pending
    for _ in range(2):
        with pytest.raises(MODULE.OriginalFinalizerPending) as caught:
            registry.drain()
        assert caught.value is pending and caught.value.actual_cause is cause


def test_reentrant_drain_is_refused():
    registry = refusal_registry()
    registry.draining = True
    with pytest.raises(MODULE.OriginalFinalizerPending):
        registry.drain()
    assert not registry.complete()


def test_wrong_process_registration_retains_but_refuses():
    registry = refusal_registry()
    registry.pid = os.getpid() + 1
    with pytest.raises(MODULE.OriginalFinalizerRefusal, match="PID_thread"):
        registry.register(MODULE.STAGES[0][0], None, None, {})
    assert registry.rejected is not None and registry.entries == {}


def test_empty_complete_is_a_diagnostic_only():
    registry = refusal_registry()
    assert registry.complete()
    diagnostic = registry.diagnostics()
    assert diagnostic["registered"] == 0
    assert diagnostic["physical_custody_authority"] is False
    assert diagnostic["disposition_settled"] is False
    assert diagnostic["acceptance_authority"] is False


def test_actual_registry_and_pending_cannot_be_pickled():
    registry = refusal_registry()
    for value in (registry, MODULE.OriginalFinalizerPending(registry, RuntimeError())):
        with pytest.raises(MODULE.OriginalFinalizerRefusal):
            pickle.dumps(value)


def test_valid_registration_is_retained_before_source_recheck():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "OriginalFinalizerContinuations")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "register")
    stores = [node.lineno for node in ast.walk(method) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Subscript)
                      and isinstance(target.value, ast.Attribute)
                      and target.value.attr == "entries" for target in node.targets)]
    checks = [node.lineno for node in ast.walk(method) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "verify_source"]
    assert len(stores) == len(checks) == 1 and stores[0] < checks[0]


def test_module_has_no_native_provider_or_connection_effect_import():
    tree = ast.parse(SOURCE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported.intersection({"subprocess", "duckdb", "sqlite3", "socket",
                                      "resource_scheduler", "docker"})
