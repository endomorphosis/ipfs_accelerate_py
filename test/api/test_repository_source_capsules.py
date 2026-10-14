"""Native capsule delivery preserves source identity, bounds and shared admission."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_capsules as capsules
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import RepositorySourceIndexError
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    IpfsDatasetsSemanticStateProvider, SemanticStateUnavailable,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceUnavailableError
from test.api.test_repository_source_index import index, scheduler, snapshot


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def test_lazy_current_native_capsule_query_and_exact_expansion(tmp_path, scheduler, monkeypatch):
    source, _package, original, _receipt, manifest = snapshot(tmp_path)
    builds = []
    build = capsules.RepositoryCapsules.__init__

    def counted(self, *args, **kwargs):
        builds.append(args[0].semantic_state.state_cid)
        build(self, *args, **kwargs)

    monkeypatch.setattr(capsules.RepositoryCapsules, "__init__", counted)
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        assert builds == []
        # The native population includes the module as well as its function;
        # the function name selects just the definition needed in this test.
        result = owner.context_query("add")
        assert len(builds) == 1
        assert result["semantic_state_cid"] == receipt["semantic_state_cid"]
        assert result["source_tree_sha256"] == receipt["source_tree_sha256"]
        assert result["ranking"] == "lexical_symbol_path"
        assert result["matching_count"] == result["selected_count"] == 1
        assert result["omitted_count"] == result["population_count"] - 1
        candidate = result["candidates"][0]
        assert candidate["name"] == "calc.add"
        assert "capsules" not in result and "admissions" not in result
        original_bytes = (source / candidate["path"]).read_bytes()
        assert candidate["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
        assert candidate["source_cid"] == cid_for_bytes(original_bytes)
        expanded = owner.capsule_expand(semantic_root_cid=result["semantic_root_cid"],
            stable_symbol_ids=[candidate["symbol_id"]])
        native = SemanticCapsule.from_dict(expanded["capsules"][0])
        assert native.capsule_cid == candidate["capsule_cid"]
        assert native.source_cid == candidate["source_cid"]
        assert native.stable_symbol_id == candidate["symbol_id"]
        assert expanded["admissions"][0]["freshness"] == "fresh"
        assert expanded["raw_source_fetch_required"][candidate["path"]] == {
            "source_sha256": candidate["source_sha256"], "source_cid": candidate["source_cid"]}
        assert len(builds) == 1
        assert owner._ingestor.stats() == receipt["ast_stats"]
        for key in ("proof_authority", "execution_authority", "completion_authority", "semantic_equivalence_claimed"):
            assert result[key] is expanded[key] is False
        assert owner._capsules.view.root.producer.repository_state_cid == receipt["semantic_state_cid"]
        assert owner._capsules.view.root.repository_id == owner.repository_id
        result["candidates"].clear()
        expanded["capsules"][0]["signature"].clear()
        assert owner.context_query("add")["candidates"] == [candidate]
        assert owner.context_query("calc add")["matching_count"] == 2


def test_query_has_explicit_no_match_and_bounded_deterministic_ranking(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path,
        files={f"module{number}.py": f"def shared{number}(): return {number}\n" for number in range(8)})
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        # The regex operates on whole identifier words: substrings do not gain a score.
        empty = owner.context_query("shared")
        assert empty["candidates"] == [] and empty["matching_count"] == empty["selected_count"] == 0
        assert empty["omitted_count"] == empty["population_count"]
        assert empty["truncated"] is False
        query = " ".join(f"shared{number}" for number in range(8))
        first = owner.context_query(query, limit=2)
        assert first == owner.context_query(query, limit=2)
        assert first["selected_count"] == 2 and first["matching_count"] == 8 and first["truncated"]
        assert [row["path"] for row in first["candidates"]] == ["module0.py", "module1.py"]
        bounded = owner.context_query(query, max_bytes=len(encoded(first)) - 1)
        assert len(encoded(bounded)) <= len(encoded(first)) - 1
        assert bounded["selected_count"] < first["selected_count"]


@pytest.mark.parametrize("kwargs", [dict(query=""), dict(query="  "), dict(query=7),
    dict(query="\ud800"), dict(query="é" * 3000), dict(query="x" * 4097),
    dict(query="calc", limit=True), dict(query="calc", limit=33),
    dict(query="calc", max_bytes=True), dict(query="calc", max_bytes=65537),
    dict(query="calc", max_bytes=1)])
def test_invalid_query_and_small_envelope_preserve_native_lookup(tmp_path, scheduler, kwargs):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        with pytest.raises(RepositorySourceIndexError, match="capsule_") as error:
            owner.context_query(**kwargs)
        assert error.value.preserves_index is True
        assert owner.symbols(query="calc.add")


@pytest.mark.parametrize("failure", ["unknown", "duplicate", "root", "size", "type"])
def test_expansion_is_all_or_error_and_optional_refusals_preserve_index(tmp_path, scheduler, failure):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        result = owner.context_query("calc add")
        identity = result["candidates"][0]["symbol_id"]
        arguments = dict(semantic_root_cid=result["semantic_root_cid"], stable_symbol_ids=[identity])
        if failure == "unknown":
            arguments["stable_symbol_ids"].append(cid_for_bytes(b"not a current native symbol"))
        elif failure == "duplicate":
            arguments["stable_symbol_ids"].append(identity)
        elif failure == "root":
            arguments["semantic_root_cid"] = cid_for_bytes(b"another native root")
        elif failure == "size":
            arguments["max_bytes"] = 1024
        else:
            arguments["stable_symbol_ids"] = identity
        with pytest.raises(RepositorySourceIndexError, match="capsule_") as error:
            owner.capsule_expand(**arguments)
        assert error.value.preserves_index is True
        assert owner.context_query("calc add")["candidates"]


def test_presentation_failures_reuse_the_verified_bundle(tmp_path, scheduler, monkeypatch):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    builds = []
    build = capsules.RepositoryCapsules.__init__

    def counted(self, *args, **kwargs):
        builds.append(True)
        build(self, *args, **kwargs)

    monkeypatch.setattr(capsules.RepositoryCapsules, "__init__", counted)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        selected = owner.context_query("add")
        arguments = dict(semantic_root_cid=selected["semantic_root_cid"],
            stable_symbol_ids=[selected["candidates"][0]["symbol_id"]])
        bundle = owner._capsules
        for changes, code in (
            ({"semantic_root_cid": cid_for_bytes(b"different root")}, "capsule_root_differs"),
            ({"stable_symbol_ids": arguments["stable_symbol_ids"] + [cid_for_bytes(b"missing")]},
             "capsule_symbol_unavailable"),
            ({"max_bytes": 1024}, "capsule_response_limit"),
        ):
            with pytest.raises(RepositorySourceIndexError, match=code) as error:
                owner.capsule_expand(**{**arguments, **changes})
            assert error.value.preserves_index is True
            assert owner._capsules is bundle
            assert owner.context_query("add") == selected
        assert owner.capsule_expand(**arguments)["capsules"]
        assert builds == [True]


def test_one_deadline_bounds_native_fences_and_capsule_admission(tmp_path, scheduler, monkeypatch):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_index as index_module
    from ipfs_datasets_py.logic.software_contracts import codebase_resources

    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        clock, observed, admitted = [100.0], [], []
        monkeypatch.setattr(index_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        observe = owner._index.observe_current
        acquire = codebase_resources.acquire_codebase_resources
        query = capsules.RepositoryCapsules.query

        def timed_observation(*args, **kwargs):
            result = observe(*args, **kwargs)
            observed.append(kwargs)
            clock[0] += 23.0 if len(observed) == 1 else 1.0
            return result

        def recorded_admission(**kwargs):
            if kwargs.get("memory_mb") == 512:
                admitted.append(kwargs)
            return acquire(**kwargs)

        def timed_query(self, *args, **kwargs):
            result = query(self, *args, **kwargs)
            clock[0] += 3.0
            return result

        monkeypatch.setattr(owner._index, "observe_current", timed_observation)
        monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", recorded_admission)
        monkeypatch.setattr(capsules.RepositoryCapsules, "query", timed_query)
        assert owner.context_query("add")["candidates"]
        assert [call["timeout_seconds"] for call in observed] == [30.0, 4.0]
        assert [call["admission_timeout_seconds"] for call in observed] == [10.0, 4.0]
        assert [call["timeout_seconds"] for call in admitted] == [7.0]


@pytest.mark.parametrize("stage", ["initial_fence", "compilation", "query", "final_fence"])
def test_elapsed_deadline_refuses_without_poisoning_current_native_state(tmp_path, scheduler, monkeypatch, stage):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_index as index_module

    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        clock, observations = [100.0], []
        observe = owner._index.observe_current
        build = capsules.RepositoryCapsules.__init__
        query = capsules.RepositoryCapsules.query
        monkeypatch.setattr(index_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

        def timed_observation(*args, **kwargs):
            result = observe(*args, **kwargs)
            observations.append(kwargs)
            if ((stage == "initial_fence" and len(observations) == 1)
                    or (stage == "final_fence" and len(observations) == 2)):
                clock[0] += 31.0
            return result

        def timed_build(self, *args, **kwargs):
            build(self, *args, **kwargs)
            if stage == "compilation":
                clock[0] += 31.0

        def timed_query(self, *args, **kwargs):
            result = query(self, *args, **kwargs)
            if stage == "query":
                clock[0] += 31.0
            return result

        monkeypatch.setattr(owner._index, "observe_current", timed_observation)
        monkeypatch.setattr(capsules.RepositoryCapsules, "__init__", timed_build)
        monkeypatch.setattr(capsules.RepositoryCapsules, "query", timed_query)
        with pytest.raises(RepositorySourceIndexError, match="capsule_timeout") as error:
            owner.context_query("add")
        assert error.value.preserves_index is True
        assert owner._capsules is None
        assert len(observations) == (2 if stage == "final_fence" else 1)
        assert owner.symbols(query="calc.add")


@pytest.mark.parametrize("bound,value,code", [("MAX_SYMBOLS", 1, "population"),
    ("MAX_SOURCE_BYTES", 1, "source"), ("MAX_STATE_BYTES", 1, "state"),
    ("MAX_BUNDLE_BYTES", 1, "bundle")])
def test_optional_compilation_bounds_do_not_break_native_symbols(tmp_path, scheduler, monkeypatch, bound, value, code):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        monkeypatch.setattr(capsules, bound, value)
        with pytest.raises(RepositorySourceIndexError, match=f"capsule_{code}_limit") as error:
            owner.context_query("calc")
        assert error.value.preserves_index is True
        assert owner._capsules is None
        assert owner.symbols(query="calc.add")


def test_capsule_provider_unavailability_is_optional(tmp_path, scheduler, monkeypatch):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)

        def unavailable(*_args, **_kwargs):
            raise SemanticStateUnavailable("build_semantic_state", "missing_exports", "unsupported producer")

        monkeypatch.setattr(IpfsDatasetsSemanticStateProvider, "_require", unavailable)
        with pytest.raises(RepositorySourceIndexError, match="capsule_provider_unavailable") as error:
            owner.context_query("calc")
        assert error.value.preserves_index is True
        assert owner.symbols(query="calc.add")


def test_capsule_admission_refusal_recovers_without_unleased_build(tmp_path, scheduler, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_resources

    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        acquire = codebase_resources.acquire_codebase_resources
        refused = []

        def pressure(**kwargs):
            if kwargs.get("memory_mb") == 512:
                refused.append(kwargs)
                raise ResourceUnavailableError("external memory pressure")
            return acquire(**kwargs)

        monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", pressure)
        with pytest.raises(RepositorySourceIndexError, match="capsule_resources_unavailable") as error:
            owner.context_query("calc")
        assert error.value.preserves_index is True and refused
        assert owner._capsules is None
        assert owner.symbols(query="calc.add")
        monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", acquire)
        assert owner.context_query("calc")["candidates"]


@pytest.mark.parametrize("when", ["before", "during_query", "during_expansion"])
@pytest.mark.parametrize("target", ["calc.py", "data.txt", ".pytest_cache/entry"])
def test_every_retained_source_is_fenced_for_query_and_expansion(tmp_path, scheduler, monkeypatch, when, target):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path,
        files={"calc.py": "def add(x): return x + 1\n", "data.txt": "data", ".pytest_cache/entry": "cache"})
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        first = owner.context_query("calc add")
        if when == "before":
            (original / target).write_text("changed")
        else:
            method = "query" if when == "during_query" else "expand"
            previous = getattr(capsules.RepositoryCapsules, method)

            def mutate(self, *args, **kwargs):
                result = previous(self, *args, **kwargs)
                (original / target).write_text("changed")
                return result

            monkeypatch.setattr(capsules.RepositoryCapsules, method, mutate)
        with pytest.raises(RepositorySourceIndexError, match="materialization_differs_from_package") as error:
            if when == "during_expansion":
                owner.capsule_expand(semantic_root_cid=first["semantic_root_cid"],
                    stable_symbol_ids=[first["candidates"][0]["symbol_id"]])
            else:
                owner.context_query("calc add")
        assert error.value.preserves_index is False
        assert owner._capsules is None
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbols()


def test_refresh_invalidates_lazy_bundle_and_old_root_without_poisoning_successor(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        before = owner.context_query("calc add")
        _source2, _package2, original2, _receipt2, manifest2 = snapshot(tmp_path, name="successor",
            files={"calc.py": "def add(value): return value + 99\n"})
        owner.refresh(original2, manifest2)
        assert owner._capsules is None
        after = owner.context_query("calc add")
        assert before["semantic_root_cid"] != after["semantic_root_cid"]
        with pytest.raises(RepositorySourceIndexError, match="capsule_root_differs") as error:
            owner.capsule_expand(semantic_root_cid=before["semantic_root_cid"],
                stable_symbol_ids=[before["candidates"][0]["symbol_id"]])
        assert error.value.preserves_index is True
        assert owner.context_query("calc add") == after


def test_capsules_never_activate_an_additional_metadata_owner(tmp_path, scheduler, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex

    calls = []
    monkeypatch.setattr(SupervisorMetaIndex, "from_env", lambda **kwargs: calls.append(kwargs))
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        calls.clear()
        result = owner.context_query("calc add")
        owner.capsule_expand(semantic_root_cid=result["semantic_root_cid"],
            stable_symbol_ids=[result["candidates"][0]["symbol_id"]])
        assert calls == []


def test_owner_failure_and_close_drop_capsule_cache(tmp_path, scheduler, monkeypatch):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        owner.context_query("calc")
        monkeypatch.setattr(owner, "_pid", -1)
        with pytest.raises(RepositorySourceIndexError, match="foreign_owner"):
            owner.context_query("calc")
        assert owner._current is owner._capsules is None
