"""Authored callback forgery control using real stdlib namespace construction."""
import importlib.machinery as machinery
import pytest
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_source_context as wrapper
from test.api.test_semantic_typed_source_context import authored_namespace_origin


def test_namespace_callback_cannot_spoof_bound_method_metadata(authored_namespace_origin):
    state = authored_namespace_origin
    expected = machinery.PathFinder._get_spec
    calls = []
    class SpoofedFinder:
        def __init__(self):
            # Instance attributes preserve exact function identity: placing this
            # function on the class would bind it and fail the older check.
            self.__self__ = expected.__self__
            self.__func__ = expected.__func__
        def __call__(self, *args):
            calls.append(args)
            raise AssertionError("A forged namespace search callback was executed")
    forged = SpoofedFinder()
    assert forged.__self__ is expected.__self__
    assert forged.__func__ is expected.__func__
    assert type(forged) is not type(expected)
    state.locations._path_finder = forged
    state.locations._last_parent_path = ()
    with pytest.raises(ValueError, match="namespace search provenance differs"):
        wrapper._typed_code_origins(state.trees, {})
    assert calls == []
