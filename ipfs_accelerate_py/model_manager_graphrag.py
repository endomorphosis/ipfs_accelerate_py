#!/usr/bin/env python3
"""
Model Manager GraphRAG Integration

This module provides a knowledge-graph layer over the model registry using
ipfs_datasets_py (when available) or a lightweight in-memory fallback.

Features:
- Represent each model as a typed entity in a knowledge graph
- Build typed relationships: derived_from, compatible_with, requires, serves
- Extract domain entities from model cards / descriptions
- Query the graph with natural-language or keyword queries
- Persist the graph to IPFS via ipfs_datasets_py content-addressed storage
- Emit graph-mutation events through DatasetsManager for provenance tracking
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ipfs_accelerate_model_manager.graphrag")

# ---------------------------------------------------------------------------
# Optional ipfs_datasets_py import
# ---------------------------------------------------------------------------
try:
    from ipfs_datasets_py.knowledge_graphs.extraction.graph import KnowledgeGraph
    from ipfs_datasets_py.knowledge_graphs.extraction.extractor import KnowledgeExtractor
    HAVE_IPFS_DATASETS_KG = True
    logger.debug("ipfs_datasets_py KnowledgeGraph available")
except ImportError:
    HAVE_IPFS_DATASETS_KG = False
    KnowledgeGraph = None
    KnowledgeExtractor = None
    logger.debug("ipfs_datasets_py KnowledgeGraph not available – using fallback graph")


# ---------------------------------------------------------------------------
# Relationship type constants
# ---------------------------------------------------------------------------
REL_DERIVED_FROM = "derived_from"       # fine-tune / distillation lineage
REL_COMPATIBLE_WITH = "compatible_with" # shared supported_backends
REL_REQUIRES = "requires"               # hardware requirements
REL_SERVES = "serves"                   # pipeline / task type
REL_MENTIONS = "mentions"               # entity extracted from model card text


# ---------------------------------------------------------------------------
# Lightweight fallback graph implementation
# ---------------------------------------------------------------------------

class _FallbackGraph:
    """
    Minimal in-memory knowledge graph used when ipfs_datasets_py is not
    installed.  Stores entities and directed labelled edges.
    """

    def __init__(self) -> None:
        # entity_id -> {"type": str, "properties": dict}
        self._entities: Dict[str, Dict[str, Any]] = {}
        # list of (source_id, relation, target_id, properties)
        self._edges: List[Tuple[str, str, str, Dict[str, Any]]] = []

    def add_entity(self, entity_id: str, entity_type: str, properties: Optional[Dict[str, Any]] = None) -> None:
        self._entities[entity_id] = {
            "type": entity_type,
            "properties": properties or {},
        }

    def add_relationship(self, source_id: str, relation: str, target_id: str,
                         properties: Optional[Dict[str, Any]] = None) -> None:
        self._edges.append((source_id, relation, target_id, properties or {}))

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self._entities.get(entity_id)

    def get_relationships(self, source_id: str, relation: Optional[str] = None
                          ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        return [
            (s, r, t, p) for s, r, t, p in self._edges
            if s == source_id and (relation is None or r == relation)
        ]

    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Simple keyword search over entity IDs and properties.
        Returns list of entity dicts whose id or properties contain the query term.
        """
        q = query.lower()
        results = []
        for eid, edata in self._entities.items():
            if q in eid.lower():
                results.append({"entity_id": eid, **edata})
                continue
            props_str = json.dumps(edata.get("properties", {})).lower()
            if q in props_str:
                results.append({"entity_id": eid, **edata})
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self._entities,
            "edges": [
                {"source": s, "relation": r, "target": t, "properties": p}
                for s, r, t, p in self._edges
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_FallbackGraph":
        g = cls()
        g._entities = data.get("entities", {})
        g._edges = [
            (e["source"], e["relation"], e["target"], e.get("properties", {}))
            for e in data.get("edges", [])
        ]
        return g


# ---------------------------------------------------------------------------
# ModelKnowledgeGraph
# ---------------------------------------------------------------------------

class ModelKnowledgeGraph:
    """
    Knowledge-graph wrapper for the model registry.

    This class builds a typed graph where:
    - Each registered model is an entity of type ``"model"``
    - Domain concepts / frameworks extracted from model cards become entities
    - Edges represent: ``derived_from``, ``compatible_with``, ``requires``,
      ``serves``, and ``mentions`` relationships

    It uses ``ipfs_datasets_py.knowledge_graphs`` when the package is
    available, and falls back to a lightweight in-memory graph otherwise.

    Parameters
    ----------
    datasets_manager:
        Optional DatasetsManager instance for event / provenance logging.
    storage:
        Optional IPFSKitStorage instance for persisting the graph to IPFS.
    """

    def __init__(
        self,
        datasets_manager: Any = None,
        storage: Any = None,
    ) -> None:
        self._datasets_manager = datasets_manager
        self._storage = storage

        if HAVE_IPFS_DATASETS_KG:
            try:
                self._graph = KnowledgeGraph()
                self._use_ipfs_datasets = True
                logger.info("ModelKnowledgeGraph using ipfs_datasets_py backend")
            except Exception as e:
                logger.warning("KnowledgeGraph init failed (%s); using fallback", e)
                self._graph = _FallbackGraph()
                self._use_ipfs_datasets = False
        else:
            self._graph = _FallbackGraph()
            self._use_ipfs_datasets = False

        # CID of the last persisted graph snapshot
        self.graph_cid: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_entity(self, entity_id: str, entity_type: str,
                    properties: Optional[Dict[str, Any]] = None) -> None:
        """Add or update a graph entity, abstracting over backend."""
        if self._use_ipfs_datasets:
            try:
                self._graph.add_entity(entity_id, entity_type=entity_type,
                                        properties=properties or {})
                return
            except Exception as e:
                logger.debug("ipfs_datasets_py add_entity failed: %s", e)
        # fallback
        assert isinstance(self._graph, _FallbackGraph)
        self._graph.add_entity(entity_id, entity_type, properties)

    def _add_relationship(self, source_id: str, relation: str, target_id: str,
                          properties: Optional[Dict[str, Any]] = None) -> None:
        """Add a directed relationship, abstracting over backend."""
        if self._use_ipfs_datasets:
            try:
                self._graph.add_relationship(source_id, relation, target_id,
                                              properties=properties or {})
                return
            except Exception as e:
                logger.debug("ipfs_datasets_py add_relationship failed: %s", e)
        assert isinstance(self._graph, _FallbackGraph)
        self._graph.add_relationship(source_id, relation, target_id, properties)

    def _emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit a graph-mutation event through DatasetsManager if available."""
        if self._datasets_manager:
            try:
                self._datasets_manager.log_event(event_name, data, level="INFO", category="GENERAL")
                self._datasets_manager.track_provenance(event_name, data)
            except Exception as e:
                logger.debug("Graph event logging failed: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_model_node(self, model_id: str, model_data: Dict[str, Any]) -> None:
        """
        Add or update a model entity in the graph.

        Parameters
        ----------
        model_id:
            Unique model identifier (e.g. ``"bert-base-uncased"``).
        model_data:
            Dict of model properties (model_name, architecture, model_type, etc.)
        """
        self._add_entity(model_id, entity_type="model", properties=model_data)

        # Extract text entities from model card / description
        text_sources = [
            model_data.get("description", ""),
            model_data.get("model_card", "") or "",
        ]
        for text in text_sources:
            if text:
                self._extract_and_link_text_entities(model_id, text)

        self._emit_event("graph_model_node_added", {
            "model_id": model_id,
            "model_type": model_data.get("model_type"),
        })
        logger.debug("Added model node: %s", model_id)

    def remove_model_node(self, model_id: str) -> None:
        """Remove a model entity from the graph."""
        # Fallback graph supports removal; ipfs_datasets_py graph may not expose it
        if isinstance(self._graph, _FallbackGraph):
            self._graph._entities.pop(model_id, None)
            self._graph._edges = [
                (s, r, t, p) for s, r, t, p in self._graph._edges
                if s != model_id and t != model_id
            ]
        self._emit_event("graph_model_node_removed", {"model_id": model_id})
        logger.debug("Removed model node: %s", model_id)

    def add_lineage_edge(self, child_model_id: str, parent_model_id: str) -> None:
        """
        Add a ``derived_from`` edge representing fine-tune / distillation lineage.
        """
        self._add_relationship(child_model_id, REL_DERIVED_FROM, parent_model_id)
        self._emit_event("graph_lineage_edge_added", {
            "child": child_model_id,
            "parent": parent_model_id,
        })

    def add_compatibility_edges(self, model_id: str, backends: List[str]) -> None:
        """
        Add ``compatible_with`` edges for each supported backend.
        """
        for backend in backends:
            backend_node = f"backend:{backend}"
            self._add_entity(backend_node, entity_type="backend",
                             properties={"name": backend})
            self._add_relationship(model_id, REL_COMPATIBLE_WITH, backend_node)

    def add_hardware_requirement_edges(self, model_id: str,
                                       hardware_requirements: Dict[str, Any]) -> None:
        """
        Add ``requires`` edges for hardware requirements.
        """
        for hw_key, hw_value in hardware_requirements.items():
            hw_node = f"hardware:{hw_key}"
            self._add_entity(hw_node, entity_type="hardware",
                             properties={"name": hw_key, "requirement": str(hw_value)})
            self._add_relationship(model_id, REL_REQUIRES, hw_node,
                                   properties={"requirement": str(hw_value)})

    def add_pipeline_edges(self, model_id: str, pipeline_types: List[str]) -> None:
        """
        Add ``serves`` edges for each supported pipeline / task type.
        """
        for pipeline in pipeline_types:
            pipeline_node = f"pipeline:{pipeline}"
            self._add_entity(pipeline_node, entity_type="pipeline",
                             properties={"name": pipeline})
            self._add_relationship(model_id, REL_SERVES, pipeline_node)

    def _extract_and_link_text_entities(self, model_id: str, text: str) -> None:
        """
        Extract domain entities from text and add ``mentions`` edges.

        Uses ipfs_datasets_py KnowledgeExtractor when available; otherwise
        applies a simple heuristic keyword scan for known frameworks, datasets,
        and domain terms.
        """
        if not text or not text.strip():
            return

        if HAVE_IPFS_DATASETS_KG and KnowledgeExtractor is not None:
            try:
                extractor = KnowledgeExtractor()
                extracted = extractor.extract(text)
                entities = extracted.get("entities", []) if isinstance(extracted, dict) else []
                for ent in entities:
                    eid = ent.get("id") or ent.get("name") or str(ent)
                    etype = ent.get("type", "concept")
                    self._add_entity(eid, entity_type=etype,
                                     properties={"name": eid, "source": "extracted"})
                    self._add_relationship(model_id, REL_MENTIONS, eid,
                                           properties={"confidence": ent.get("confidence", 1.0)})
                return
            except Exception as e:
                logger.debug("KnowledgeExtractor failed: %s", e)

        # Heuristic fallback: scan for known framework and dataset keywords
        KNOWN_ENTITIES: Dict[str, str] = {
            "pytorch": "framework",
            "tensorflow": "framework",
            "jax": "framework",
            "onnx": "framework",
            "triton": "framework",
            "vllm": "framework",
            "tgi": "framework",
            "huggingface": "platform",
            "transformers": "library",
            "diffusers": "library",
            "peft": "library",
            "imagenet": "dataset",
            "squad": "dataset",
            "glue": "dataset",
            "superglue": "dataset",
            "openwebtext": "dataset",
            "the pile": "dataset",
            "llama": "model_family",
            "mistral": "model_family",
            "falcon": "model_family",
            "gpt": "model_family",
            "bert": "model_family",
            "t5": "model_family",
            "cuda": "hardware",
            "rocm": "hardware",
            "mps": "hardware",
            "openvino": "hardware",
            "tensorrt": "hardware",
        }

        text_lower = text.lower()
        for keyword, entity_type in KNOWN_ENTITIES.items():
            if keyword in text_lower:
                node_id = f"{entity_type}:{keyword}"
                self._add_entity(node_id, entity_type=entity_type,
                                 properties={"name": keyword, "source": "heuristic"})
                self._add_relationship(model_id, REL_MENTIONS, node_id,
                                       properties={"confidence": 0.7})

    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.

        Delegates to ipfs_datasets_py's query layer when available, otherwise
        performs keyword search over entity IDs and properties.

        Parameters
        ----------
        query:
            Natural-language query string, e.g.
            ``"models fine-tuned from llama-3 that support cuda"``.

        Returns
        -------
        List of result dicts with at minimum ``entity_id`` and ``type`` fields.
        """
        if self._use_ipfs_datasets:
            try:
                # ipfs_datasets_py graph may expose a query() method
                if hasattr(self._graph, "query"):
                    results = self._graph.query(query)
                    if isinstance(results, list):
                        return results
            except Exception as e:
                logger.debug("ipfs_datasets_py graph query failed: %s", e)

        # Fallback to keyword search
        if isinstance(self._graph, _FallbackGraph):
            return self._graph.query(query)

        return []

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the graph to a plain dict."""
        if isinstance(self._graph, _FallbackGraph):
            return self._graph.to_dict()
        # ipfs_datasets_py graph
        try:
            if hasattr(self._graph, "to_dict"):
                return self._graph.to_dict()
        except Exception:
            pass
        return {}

    def persist_to_ipfs(self) -> Optional[str]:
        """
        Persist the current graph snapshot to IPFS via IPFSKitStorage.

        Returns the CID of the persisted graph, or None on failure.
        """
        if not self._storage:
            return None
        try:
            data = json.dumps(self.to_dict(), default=str)
            cid = self._storage.store(data, filename="model_knowledge_graph.json", pin=True)
            self.graph_cid = cid
            logger.info("Persisted model knowledge graph to IPFS: %s", cid)
            return cid
        except Exception as e:
            logger.warning("Failed to persist knowledge graph: %s", e)
            return None

    def load_from_ipfs(self, cid: str) -> bool:
        """
        Load a graph snapshot from IPFS by CID.

        Returns True on success.
        """
        if not self._storage:
            return False
        try:
            payload = self._storage.retrieve(cid)
            if payload is None:
                return False
            data = json.loads(payload if isinstance(payload, str) else payload.decode())
            self._graph = _FallbackGraph.from_dict(data)
            self._use_ipfs_datasets = False  # loaded as fallback dict
            self.graph_cid = cid
            logger.info("Loaded model knowledge graph from IPFS: %s", cid)
            return True
        except Exception as e:
            logger.warning("Failed to load knowledge graph from IPFS: %s", e)
            return False
