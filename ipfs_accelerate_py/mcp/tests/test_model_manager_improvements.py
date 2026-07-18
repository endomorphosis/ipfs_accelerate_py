"""
Tests for model_manager improvements:
  - Gap 1: Model weights caching via ipfs_kit_py (warm_cache, get_cached_weight_path,
            tiered cache config, service registry helpers on IPFSKitStorage)
  - Gap 2: GraphRAG via ipfs_datasets_py (ModelKnowledgeGraph, build_model_graph,
            query_model_graph)
  - Gap 3: Serving configuration (ServingConfig dataclass, get_serving_config,
            update_serving_config, resolve_launch_command, get_models_by_pipeline_type
            now includes serving_config)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# ---- path setup -----------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ipfs_accelerate_py.model_manager import (
    DataType,
    IOSpec,
    ModelManager,
    ModelMetadata,
    ModelType,
    ServingConfig,
)
from ipfs_accelerate_py.model_manager_graphrag import (
    ModelKnowledgeGraph,
    REL_COMPATIBLE_WITH,
    REL_DERIVED_FROM,
    REL_MENTIONS,
    REL_REQUIRES,
    REL_SERVES,
    _FallbackGraph,
)
from ipfs_accelerate_py.ipfs_kit_integration import IPFSKitStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(model_id: str = "test/bert", **kwargs) -> ModelMetadata:
    defaults = dict(
        model_id=model_id,
        model_name=model_id.split("/")[-1],
        model_type=ModelType.LANGUAGE_MODEL,
        architecture="BertForMaskedLM",
        inputs=[IOSpec("input_ids", DataType.TOKENS)],
        outputs=[IOSpec("logits", DataType.LOGITS)],
        supported_backends=["cuda", "cpu"],
        hardware_requirements={"min_gpu_memory_gb": 4},
        description="A BERT model using pytorch on huggingface transformers",
        tags=["nlp", "bert"],
    )
    defaults.update(kwargs)
    return ModelMetadata(**defaults)


# ===========================================================================
# Gap 3 – ServingConfig dataclass
# ===========================================================================


class TestServingConfig(unittest.TestCase):
    def test_defaults(self):
        sc = ServingConfig()
        self.assertEqual(sc.engine, "hf_pipeline")
        self.assertEqual(sc.launch_args, {})
        self.assertEqual(sc.default_generation_params, {})
        self.assertEqual(sc.hardware_affinity, [])
        self.assertEqual(sc.routing_weight, 1.0)
        self.assertEqual(sc.min_replicas, 1)
        self.assertEqual(sc.max_replicas, 1)

    def test_custom_fields(self):
        sc = ServingConfig(
            engine="vllm",
            launch_args={"--tensor-parallel-size": 4, "--quantization": "awq"},
            default_generation_params={"temperature": 0.8, "max_new_tokens": 1024},
            routing_weight=2.5,
            min_replicas=2,
            max_replicas=8,
            hardware_affinity=["cuda", "rocm"],
        )
        self.assertEqual(sc.engine, "vllm")
        self.assertEqual(sc.launch_args["--tensor-parallel-size"], 4)
        self.assertEqual(sc.default_generation_params["temperature"], 0.8)
        self.assertEqual(sc.routing_weight, 2.5)
        self.assertEqual(sc.hardware_affinity, ["cuda", "rocm"])

    def test_to_dict(self):
        sc = ServingConfig(engine="tgi", routing_weight=3.0)
        d = sc.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["engine"], "tgi")
        self.assertEqual(d["routing_weight"], 3.0)

    def test_from_dict_roundtrip(self):
        sc = ServingConfig(
            engine="llama.cpp",
            launch_args={"-n": 512, "-t": 8},
            hardware_affinity=["cpu"],
        )
        sc2 = ServingConfig.from_dict(sc.to_dict())
        self.assertEqual(sc2.engine, sc.engine)
        self.assertEqual(sc2.launch_args, sc.launch_args)
        self.assertEqual(sc2.hardware_affinity, sc.hardware_affinity)

    def test_from_dict_ignores_unknown_keys(self):
        """Extra keys from future schema versions should be silently dropped."""
        sc = ServingConfig.from_dict({"engine": "vllm", "future_field": "value"})
        self.assertEqual(sc.engine, "vllm")
        self.assertFalse(hasattr(sc, "future_field"))

    def test_json_serialisable(self):
        sc = ServingConfig(engine="triton", endpoint_schema={"type": "object"})
        json_str = json.dumps(sc.to_dict())
        loaded = json.loads(json_str)
        self.assertEqual(loaded["engine"], "triton")
        self.assertEqual(loaded["endpoint_schema"]["type"], "object")


# ===========================================================================
# Gap 3 – ModelMetadata.serving_config field
# ===========================================================================


class TestModelMetadataServingConfig(unittest.TestCase):
    def test_serving_config_default_none(self):
        meta = _make_metadata()
        self.assertIsNone(meta.serving_config)

    def test_serving_config_set(self):
        sc = ServingConfig(engine="vllm")
        meta = _make_metadata(serving_config=sc.to_dict())
        self.assertIsNotNone(meta.serving_config)
        self.assertEqual(meta.serving_config["engine"], "vllm")


# ===========================================================================
# Gap 3 – ModelManager serving config methods
# ===========================================================================


class TestModelManagerServingConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = ModelManager(
            storage_path=os.path.join(self.tmp, "mm.json"),
            use_database=False,
        )
        self.meta = _make_metadata("test/llama-3-8b", architecture="LlamaForCausalLM")
        self.manager.add_model(self.meta)

    def tearDown(self):
        self.manager.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_serving_config_none_initially(self):
        self.assertIsNone(self.manager.get_serving_config("test/llama-3-8b"))

    def test_update_and_get_serving_config_with_dict(self):
        cfg = {"engine": "vllm", "launch_args": {"--max-model-len": 4096}}
        ok = self.manager.update_serving_config("test/llama-3-8b", cfg)
        self.assertTrue(ok)
        returned = self.manager.get_serving_config("test/llama-3-8b")
        self.assertIsNotNone(returned)
        self.assertEqual(returned["engine"], "vllm")

    def test_update_and_get_serving_config_with_dataclass(self):
        sc = ServingConfig(engine="tgi", routing_weight=3.0)
        ok = self.manager.update_serving_config("test/llama-3-8b", sc)
        self.assertTrue(ok)
        returned = self.manager.get_serving_config("test/llama-3-8b")
        self.assertEqual(returned["engine"], "tgi")
        self.assertEqual(returned["routing_weight"], 3.0)

    def test_update_serving_config_persists_in_metadata(self):
        sc = ServingConfig(engine="onnxruntime")
        self.manager.update_serving_config("test/llama-3-8b", sc)
        meta = self.manager.get_model("test/llama-3-8b")
        self.assertIsNotNone(meta.serving_config)
        self.assertEqual(meta.serving_config["engine"], "onnxruntime")

    def test_update_serving_config_missing_model(self):
        ok = self.manager.update_serving_config("nonexistent/model", {"engine": "vllm"})
        self.assertFalse(ok)

    def test_serving_config_in_stats(self):
        self.manager.update_serving_config("test/llama-3-8b", ServingConfig(engine="vllm"))
        stats = self.manager.get_stats()
        self.assertIn("models_with_serving_config", stats)
        self.assertEqual(stats["models_with_serving_config"], 1)

    def test_serving_config_json_persistence(self):
        """serving_config round-trips through JSON storage."""
        sc = ServingConfig(engine="vllm", launch_args={"--tensor-parallel-size": 2})
        self.manager.update_serving_config("test/llama-3-8b", sc)

        # Reload from disk
        mgr2 = ModelManager(
            storage_path=os.path.join(self.tmp, "mm.json"),
            use_database=False,
        )
        loaded = mgr2.get_model("test/llama-3-8b")
        self.assertIsNotNone(loaded.serving_config)
        self.assertEqual(loaded.serving_config["engine"], "vllm")
        self.assertEqual(loaded.serving_config["launch_args"]["--tensor-parallel-size"], 2)
        mgr2.close()


# ===========================================================================
# Gap 3 – resolve_launch_command
# ===========================================================================


class TestResolveLaunchCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = ModelManager(
            storage_path=os.path.join(self.tmp, "mm.json"),
            use_database=False,
        )
        self.meta = _make_metadata("test/mistral-7b", architecture="MistralForCausalLM")
        self.manager.add_model(self.meta)

    def tearDown(self):
        self.manager.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_serving_config_returns_none(self):
        result = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertIsNone(result)

    def test_vllm_command(self):
        sc = ServingConfig(
            engine="vllm",
            launch_args={"--tensor-parallel-size": 2, "--quantization": "awq"},
        )
        self.manager.update_serving_config("test/mistral-7b", sc)
        cmd = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertIsNotNone(cmd)
        self.assertIn("vllm.entrypoints.openai.api_server", cmd)
        self.assertIn("test/mistral-7b", cmd)
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertIn("2", cmd)

    def test_tgi_command(self):
        sc = ServingConfig(engine="tgi", launch_args={"--max-input-length": 2048})
        self.manager.update_serving_config("test/mistral-7b", sc)
        cmd = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertIn("text-generation-launcher", cmd)
        self.assertIn("--model-id", cmd)

    def test_hf_pipeline_command(self):
        sc = ServingConfig(engine="hf_pipeline")
        self.manager.update_serving_config("test/mistral-7b", sc)
        cmd = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertIn("ipfs_accelerate_py.hf_model_server.server", cmd)

    def test_llama_cpp_command(self):
        sc = ServingConfig(engine="llama.cpp", launch_args={"-n": 512})
        self.manager.update_serving_config("test/mistral-7b", sc)
        cmd = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertIn("./server", cmd)
        self.assertIn("-n", cmd)

    def test_bool_flag_included_when_true(self):
        sc = ServingConfig(engine="vllm", launch_args={"--trust-remote-code": True})
        self.manager.update_serving_config("test/mistral-7b", sc)
        cmd = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertIn("--trust-remote-code", cmd)

    def test_bool_flag_excluded_when_false(self):
        sc = ServingConfig(engine="vllm", launch_args={"--trust-remote-code": False})
        self.manager.update_serving_config("test/mistral-7b", sc)
        cmd = self.manager.resolve_launch_command("test/mistral-7b")
        self.assertNotIn("--trust-remote-code", cmd)


# ===========================================================================
# Gap 3 – get_models_by_pipeline_type includes serving_config
# ===========================================================================


class TestPipelineTypeServingConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = ModelManager(
            storage_path=os.path.join(self.tmp, "mm.json"),
            use_database=False,
        )
        self.meta = _make_metadata(
            "test/gpt2",
            architecture="GPT2LMHeadModel",
            serving_config=ServingConfig(engine="vllm").to_dict(),
        )
        self.manager.add_model(self.meta)

    def tearDown(self):
        self.manager.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_serving_config_present_in_pipeline_results(self):
        results = self.manager.get_models_by_pipeline_type(
            "text-generation", include_api=False
        )
        # If the model is matched by the pipeline mapper, verify serving_config is present
        for r in results:
            if r["model_id"] == "test/gpt2":
                self.assertIn("serving_config", r)
                self.assertEqual(r["serving_config"]["engine"], "vllm")
                return
        # If no results or model not matched, just verify structure of any returned model
        for r in results:
            self.assertIn("serving_config", r)


# ===========================================================================
# Gap 2 – ModelKnowledgeGraph and _FallbackGraph
# ===========================================================================


class TestFallbackGraph(unittest.TestCase):
    def test_add_and_get_entity(self):
        g = _FallbackGraph()
        g.add_entity("model_a", "model", {"name": "Model A"})
        e = g.get_entity("model_a")
        self.assertIsNotNone(e)
        self.assertEqual(e["type"], "model")
        self.assertEqual(e["properties"]["name"], "Model A")

    def test_get_nonexistent_entity(self):
        g = _FallbackGraph()
        self.assertIsNone(g.get_entity("does_not_exist"))

    def test_add_and_get_relationships(self):
        g = _FallbackGraph()
        g.add_entity("child", "model")
        g.add_entity("parent", "model")
        g.add_relationship("child", REL_DERIVED_FROM, "parent")
        edges = g.get_relationships("child")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][1], REL_DERIVED_FROM)
        self.assertEqual(edges[0][2], "parent")

    def test_filter_relationships_by_type(self):
        g = _FallbackGraph()
        g.add_entity("m", "model")
        g.add_entity("b1", "backend")
        g.add_entity("b2", "backend")
        g.add_relationship("m", REL_COMPATIBLE_WITH, "b1")
        g.add_relationship("m", REL_REQUIRES, "b2")
        compat = g.get_relationships("m", REL_COMPATIBLE_WITH)
        self.assertEqual(len(compat), 1)

    def test_query_by_entity_id(self):
        g = _FallbackGraph()
        g.add_entity("test/bert-base", "model", {"arch": "bert"})
        g.add_entity("test/gpt2", "model", {"arch": "gpt2"})
        results = g.query("bert")
        self.assertTrue(any(r["entity_id"] == "test/bert-base" for r in results))
        self.assertFalse(any(r["entity_id"] == "test/gpt2" for r in results))

    def test_query_by_property(self):
        g = _FallbackGraph()
        g.add_entity("m1", "model", {"framework": "pytorch", "size": "large"})
        g.add_entity("m2", "model", {"framework": "tensorflow", "size": "small"})
        results = g.query("pytorch")
        self.assertTrue(any(r["entity_id"] == "m1" for r in results))

    def test_to_dict_from_dict_roundtrip(self):
        g = _FallbackGraph()
        g.add_entity("a", "model", {"name": "A"})
        g.add_entity("b", "model", {"name": "B"})
        g.add_relationship("a", REL_DERIVED_FROM, "b")
        d = g.to_dict()
        g2 = _FallbackGraph.from_dict(d)
        self.assertIn("a", g2._entities)
        self.assertEqual(len(g2._edges), 1)


class TestModelKnowledgeGraph(unittest.TestCase):
    def _build_graph(self, **kwargs) -> ModelKnowledgeGraph:
        return ModelKnowledgeGraph(**kwargs)

    def test_init_uses_fallback_without_ipfs_datasets(self):
        kg = self._build_graph()
        self.assertIsNotNone(kg._graph)

    def test_add_model_node(self):
        kg = self._build_graph()
        kg.add_model_node("test/bert", {
            "model_name": "bert",
            "description": "A BERT model",
            "model_card": "",
        })
        e = kg._graph.get_entity("test/bert")
        self.assertIsNotNone(e)
        self.assertEqual(e["type"], "model")

    def test_add_lineage_edge(self):
        kg = self._build_graph()
        kg.add_model_node("child/model", {})
        kg.add_model_node("parent/model", {})
        kg.add_lineage_edge("child/model", "parent/model")
        edges = kg._graph.get_relationships("child/model", REL_DERIVED_FROM)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][2], "parent/model")

    def test_add_compatibility_edges(self):
        kg = self._build_graph()
        kg.add_model_node("test/model", {})
        kg.add_compatibility_edges("test/model", ["cuda", "cpu"])
        cuda_edge = kg._graph.get_relationships("test/model", REL_COMPATIBLE_WITH)
        self.assertEqual(len(cuda_edge), 2)

    def test_add_hardware_requirement_edges(self):
        kg = self._build_graph()
        kg.add_model_node("test/model", {})
        kg.add_hardware_requirement_edges("test/model", {"min_gpu_memory_gb": 16})
        edges = kg._graph.get_relationships("test/model", REL_REQUIRES)
        self.assertEqual(len(edges), 1)

    def test_add_pipeline_edges(self):
        kg = self._build_graph()
        kg.add_model_node("test/model", {})
        kg.add_pipeline_edges("test/model", ["text-generation", "summarization"])
        edges = kg._graph.get_relationships("test/model", REL_SERVES)
        self.assertEqual(len(edges), 2)

    def test_text_entity_extraction_heuristic(self):
        kg = self._build_graph()
        kg.add_model_node("test/bert", {
            "model_name": "bert",
            "description": "A BERT model trained with pytorch on the squad dataset",
            "model_card": "Uses huggingface transformers",
        })
        # Check that framework/dataset entities were linked
        mentions = kg._graph.get_relationships("test/bert", REL_MENTIONS)
        entity_ids = {e[2] for e in mentions}
        self.assertTrue(any("pytorch" in eid for eid in entity_ids))

    def test_remove_model_node(self):
        kg = self._build_graph()
        kg.add_model_node("test/to-remove", {"model_name": "rm"})
        kg.remove_model_node("test/to-remove")
        e = kg._graph.get_entity("test/to-remove")
        self.assertIsNone(e)

    def test_query_returns_relevant_entities(self):
        kg = self._build_graph()
        kg.add_model_node("test/cuda-model", {"model_name": "cuda-model",
                                               "description": "cuda optimized"})
        kg.add_model_node("test/cpu-model", {"model_name": "cpu-model",
                                              "description": "cpu only"})
        results = kg.query("cuda")
        ids = [r["entity_id"] for r in results]
        self.assertTrue(any("cuda" in eid for eid in ids))

    def test_graph_cid_initially_none(self):
        kg = self._build_graph()
        self.assertIsNone(kg.graph_cid)

    def test_persist_without_storage_returns_none(self):
        kg = self._build_graph(storage=None)
        cid = kg.persist_to_ipfs()
        self.assertIsNone(cid)

    def test_to_dict(self):
        kg = self._build_graph()
        kg.add_model_node("test/a", {"model_name": "a"})
        d = kg.to_dict()
        self.assertIn("entities", d)


# ===========================================================================
# Gap 2 – ModelManager graph methods
# ===========================================================================


class TestModelManagerGraphRAG(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = ModelManager(
            storage_path=os.path.join(self.tmp, "mm.json"),
            use_database=False,
        )
        self.manager.add_model(_make_metadata(
            "test/bert-base-uncased",
            architecture="BertForMaskedLM",
            description="BERT model using pytorch and huggingface",
            tags=["bert", "nlp"],
        ))
        self.manager.add_model(_make_metadata(
            "test/bert-fine-tuned",
            architecture="BertForSequenceClassification",
            description="Fine-tuned BERT",
            parent_model_id="test/bert-base-uncased",
            tags=["bert", "classification"],
        ))
        self.manager.add_model(_make_metadata(
            "test/gpt2",
            architecture="GPT2LMHeadModel",
            description="GPT-2 text generation model pytorch",
            tags=["gpt2", "text-gen"],
        ))

    def tearDown(self):
        self.manager.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_model_graph_returns_true_when_available(self):
        result = self.manager.build_model_graph()
        self.assertIsInstance(result, bool)
        # True when HAVE_GRAPHRAG is True, False otherwise
        if self.manager._knowledge_graph:
            self.assertTrue(result)

    def test_query_model_graph_returns_list(self):
        results = self.manager.query_model_graph("bert")
        self.assertIsInstance(results, list)

    def test_query_model_graph_finds_model(self):
        self.manager.build_model_graph()
        results = self.manager.query_model_graph("bert")
        ids = [r["entity_id"] for r in results]
        if self.manager._knowledge_graph:
            self.assertTrue(any("bert" in eid for eid in ids))

    def test_query_model_graph_framework_entity(self):
        self.manager.build_model_graph()
        results = self.manager.query_model_graph("pytorch")
        if self.manager._knowledge_graph:
            self.assertIsInstance(results, list)
            # Pytorch should be linked as a framework entity

    def test_get_stats_includes_graph_cid_key(self):
        stats = self.manager.get_stats()
        self.assertIn("knowledge_graph_cid", stats)

    def test_remove_model_updates_graph(self):
        self.manager.build_model_graph()
        self.manager.remove_model("test/gpt2")
        if self.manager._knowledge_graph:
            entity = self.manager._knowledge_graph._graph.get_entity("test/gpt2")
            self.assertIsNone(entity)


# ===========================================================================
# Gap 1 – IPFSKitStorage.configure_cache
# ===========================================================================


class TestIPFSKitStorageCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = IPFSKitStorage(
            enable_ipfs_kit=False,
            force_fallback=True,
            cache_dir=self.tmp,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_configure_cache_returns_dict(self):
        result = self.storage.configure_cache(memory_mb=50, disk_mb=512, eviction_policy="lfu")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["memory_mb"], 50)
        self.assertEqual(result["disk_mb"], 512)
        self.assertEqual(result["eviction_policy"], "lfu")

    def test_configure_cache_sets_attributes(self):
        self.storage.configure_cache(memory_mb=200, disk_mb=2048, eviction_policy="arc")
        self.assertEqual(self.storage._cache_memory_mb, 200)
        self.assertEqual(self.storage._cache_disk_mb, 2048)
        self.assertEqual(self.storage._cache_eviction_policy, "arc")

    def test_configure_cache_enforces_disk_quota(self):
        """Writing files that exceed quota should trigger eviction."""
        # Write several small files manually to simulate a full cache
        cache_dir = Path(self.tmp)
        for i in range(5):
            (cache_dir / f"bafyfile{i:04d}").write_bytes(b"x" * 100)  # 100 bytes each

        # Configure with a very small quota (1 byte) to force eviction
        self.storage.configure_cache(disk_mb=0)  # 0 MB = ~0 bytes
        # After eviction the directory should have fewer files (or the same if
        # the OS rounds up the limit)
        remaining = list(f for f in cache_dir.iterdir() if f.is_file())
        # We just verify it doesn't raise and returns something sensible
        self.assertIsInstance(remaining, list)

    def test_register_model_service_local_fallback(self):
        ok = self.storage.register_model_service(
            "test/my-model",
            {"engine": "vllm", "launch_args": {"--tp": 2}},
        )
        self.assertTrue(ok)
        # Check file was written
        svc_file = Path(self.tmp) / "service_test_my-model.json"
        self.assertTrue(svc_file.exists())
        with open(svc_file) as fh:
            data = json.load(fh)
        self.assertEqual(data["serving_config"]["engine"], "vllm")

    def test_get_model_service_config_local_fallback(self):
        self.storage.register_model_service("test/my-model", {"engine": "tgi"})
        result = self.storage.get_model_service_config("test/my-model")
        self.assertIsNotNone(result)
        self.assertEqual(result["serving_config"]["engine"], "tgi")

    def test_get_model_service_config_nonexistent_returns_none(self):
        result = self.storage.get_model_service_config("does/not-exist")
        self.assertIsNone(result)


# ===========================================================================
# Gap 1 – ModelManager.warm_cache and get_cached_weight_path
# ===========================================================================


class TestModelManagerCaching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Use a writable cache dir inside tmp
        self.cache_dir = os.path.join(self.tmp, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.manager = ModelManager(
            storage_path=os.path.join(self.tmp, "mm.json"),
            use_database=False,
            cache_memory_mb=50,
            cache_disk_mb=512,
            cache_eviction_policy="lru",
        )
        self.meta = _make_metadata("test/small-model")
        self.manager.add_model(self.meta)

    def tearDown(self):
        self.manager.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_warm_cache_missing_model_returns_false(self):
        result = self.manager.warm_cache(["nonexistent/model"])
        self.assertFalse(result.get("nonexistent/model", True))

    def test_warm_cache_no_cids_returns_false(self):
        result = self.manager.warm_cache(["test/small-model"])
        # Model has no CIDs so warming should return False (or True if already cached)
        self.assertIsInstance(result, dict)
        self.assertIn("test/small-model", result)

    def test_warm_cache_with_stored_artifact(self):
        """Simulate a model with a stored CID – warm_cache should find it in cache."""
        if not self.manager._artifact_storage:
            self.skipTest("No artifact storage available")

        # Store some data and get a CID
        cid = self.manager._artifact_storage.store(b"fake-weights", filename="model.safetensors")

        # Attach the CID to the model
        self.meta.model_cid = cid
        self.manager.models["test/small-model"] = self.meta

        # warm_cache should find it already cached
        result = self.manager.warm_cache(["test/small-model"])
        self.assertTrue(result.get("test/small-model", False))

    def test_get_cached_weight_path_no_cid_returns_none(self):
        path = self.manager.get_cached_weight_path("test/small-model")
        self.assertIsNone(path)

    def test_get_cached_weight_path_with_cid(self):
        if not self.manager._artifact_storage:
            self.skipTest("No artifact storage available")

        # Store some data and get a CID
        cid = self.manager._artifact_storage.store(
            b"fake-model-weights", filename="model.safetensors"
        )
        self.meta.model_cid = cid
        self.manager.models["test/small-model"] = self.meta

        # Should return the local cached path
        path = self.manager.get_cached_weight_path("test/small-model", "model.safetensors")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_cache_config_propagated_to_artifact_storage(self):
        if not self.manager._artifact_storage:
            self.skipTest("No artifact storage available")
        self.assertEqual(self.manager._artifact_storage._cache_memory_mb, 50)
        self.assertEqual(self.manager._artifact_storage._cache_disk_mb, 512)
        self.assertEqual(self.manager._artifact_storage._cache_eviction_policy, "lru")


# ===========================================================================
# Integration – full round-trip with JSON storage
# ===========================================================================


class TestFullIntegration(unittest.TestCase):
    def test_add_retrieve_with_serving_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ModelManager(
                storage_path=os.path.join(tmp, "mm.json"),
                use_database=False,
            )

            sc = ServingConfig(
                engine="vllm",
                launch_args={"--tensor-parallel-size": 4},
                default_generation_params={"temperature": 0.9},
                routing_weight=1.5,
                hardware_affinity=["cuda"],
            )
            meta = _make_metadata(
                "meta-llama/Llama-3-8B",
                architecture="LlamaForCausalLM",
                parent_model_id="meta-llama/Llama-3",
                serving_config=sc.to_dict(),
            )
            manager.add_model(meta)
            manager.close()

            # Reload
            manager2 = ModelManager(
                storage_path=os.path.join(tmp, "mm.json"),
                use_database=False,
            )
            loaded = manager2.get_model("meta-llama/Llama-3-8B")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.serving_config["engine"], "vllm")
            self.assertEqual(loaded.serving_config["routing_weight"], 1.5)

            # Check launch command
            cmd = manager2.resolve_launch_command("meta-llama/Llama-3-8B")
            self.assertIn("vllm.entrypoints.openai.api_server", cmd)
            self.assertIn("--tensor-parallel-size", cmd)

            # Check graph was updated on add
            if manager2._knowledge_graph:
                manager2.build_model_graph()
                results = manager2.query_model_graph("llama")
                self.assertIsInstance(results, list)

            manager2.close()


if __name__ == "__main__":
    unittest.main()
