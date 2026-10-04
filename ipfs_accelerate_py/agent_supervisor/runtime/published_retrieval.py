"""Closed local lexical retrieval rebuild policy for admitted publication.

This reproduces the explicit TF-IDF symbol-name lane. A learned predecessor
cannot select it, and a changed vocabulary is unavailable rather than silently
changing the pinned dimensional/configuration contract.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import PurePosixPath
import re

from ..analysis.code_symbol_vector_index import (
    build_code_symbol_vector_index, search_code_symbol_vector_index,
)
from ..analysis.program_ast_adapters import build_program_evidence_index
from ..proof.formal_verification_contracts import content_identity
from . import code_retrieval_context as retrieval
from .task_context_bundle import load_task_context_nomination


POLICY = "lexical-tfidf-symbols@1"
SCHEMA = "supervisor-published-lexical-retrieval-policy@1"


class RetrievalRefreshUnavailable(ValueError):
    """An optional pinned lane cannot produce a current replacement."""

    def __init__(self, reason):
        if reason not in {"lexical_vocabulary_changed", "query_has_no_vocabulary_terms"}:
            raise ValueError("unknown retrieval availability reason")
        self.reason = reason
        super().__init__(reason)


def _configuration(symbols):
    docs = {symbol: Counter(re.findall("[a-z0-9]+", symbol.lower())) for symbol in symbols}
    if not docs:
        raise RetrievalRefreshUnavailable("lexical_vocabulary_changed")
    vocabulary = sorted({term for doc in docs.values() for term in doc})
    weights = {term: 1 + math.log((1 + len(docs)) / (1 + sum(term in doc for doc in docs.values())))
               for term in vocabulary}
    config = content_identity({"vocabulary": vocabulary,
                               "weights": {term: value.hex() for term, value in weights.items()}})
    return vocabulary, weights, config


def _vector(text, vocabulary, weights):
    words = Counter(re.findall("[a-z0-9]+", text.lower()))
    values = [words[term] * weights[term] for term in vocabulary]
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        raise RetrievalRefreshUnavailable("query_has_no_vocabulary_terms")
    return tuple(value / norm for value in values)


def _verify_lexical_snapshot(snapshot):
    vocabulary, weights, config = _configuration(row.qualified_symbol for row in snapshot.rows)
    pin = snapshot.config
    if (pin.model_id != POLICY or pin.model_revision != "1"
            or pin.producer_id != "code-symbol-vector-indexer@1"
            or pin.chunker_id != "ast-symbol@1" or pin.normalization != "l2"
            or pin.metric != "cosine" or pin.configuration_id != config
            or pin.dimensions != len(vocabulary)
            or any(row.embedding != _vector(row.qualified_symbol, vocabulary, weights) for row in snapshot.rows)):
        raise ValueError("predecessor is not the exact native lexical TF-IDF policy")
    return vocabulary, weights, config


def bind_published_retrieval_policy(*, repository, bundle, task_cid, task_id):
    """Bind the reviewed policy to current, native initial retrieval objects."""
    from .published_task_context import _previous_retrieval
    metadata = load_task_context_nomination(repository=repository,
        artifact=bundle["artifact"], expected_sha256=bundle["sha256"], task_cid=task_cid, task_id=task_id)
    previous = _previous_retrieval(repository, metadata, task_id)
    if previous is None or previous[0]["status"] != "current":
        raise ValueError("published lexical policy requires current initial retrieval")
    context, snapshot, result = previous
    _verify_lexical_snapshot(snapshot)
    return {"schema": SCHEMA, "policy": POLICY,
        "task_cid": task_cid, "task_id": task_id,
        "artifact": metadata["code retrieval artifact"], "sha256": metadata["code retrieval sha256"],
        "index_id": snapshot.index_id, "config_id": snapshot.config.config_id,
        "query_id": result.query.query_id,
        "query_sha256": hashlib.sha256(context["query_text"].encode()).hexdigest(),
        "scope_paths": list(snapshot.included_paths), "completion_authority": False,
        "execution_authority": False, "learned_embeddings": False}


def published_retrieval_rebuilder(*, repository, bundle, binding):
    """Return the built-in callback after immutable nomination revalidation."""
    from .published_task_context import _previous_retrieval
    required = {"schema", "policy", "task_cid", "task_id", "artifact", "sha256", "index_id",
                "config_id", "query_id", "query_sha256", "scope_paths", "completion_authority",
                "execution_authority", "learned_embeddings"}
    if (not isinstance(binding, dict) or set(binding) != required or binding["schema"] != SCHEMA
            or binding["policy"] != POLICY or any(binding[name] is not False for name in (
                "completion_authority", "execution_authority", "learned_embeddings"))):
        raise ValueError("invalid published lexical retrieval policy")
    metadata = load_task_context_nomination(repository=repository, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=binding["task_cid"], task_id=binding["task_id"])
    previous = _previous_retrieval(repository, metadata, binding["task_id"])
    if previous is None:
        raise ValueError("published lexical predecessor disappeared")
    context, original, original_result = previous
    if (metadata["code retrieval artifact"] != binding["artifact"]
            or metadata["code retrieval sha256"] != binding["sha256"]
            or original.index_id != binding["index_id"]
            or original.config.config_id != binding["config_id"]
            or original_result.query.query_id != binding["query_id"]
            or list(original.included_paths) != binding["scope_paths"]
            or hashlib.sha256(context["query_text"].encode()).hexdigest() != binding["query_sha256"]):
        raise ValueError("published lexical predecessor binding changed")
    old_vocabulary, old_weights, old_config = _verify_lexical_snapshot(original)

    def rebuild(*, repository, previous_snapshot, previous_result, query_text, output):
        if (previous_snapshot != original or previous_result != original_result
                or query_text != context["query_text"]):
            raise ValueError("lexical rebuild differs from its signed predecessor/query")
        sources, hashes = retrieval._sources(repository, original.included_paths)
        evidence = build_program_evidence_index(sources)
        if (not evidence.exhaustive or any(item.status != "success" for item in evidence.results)
                or set(evidence.ast_index.paths) != set(original.included_paths)):
            raise ValueError("published lexical source scan is incomplete")
        symbols = []
        for indexed in evidence.ast_index.path_records:
            parts = list(PurePosixPath(indexed.path.removesuffix(".py")).parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            module = ".".join(parts)
            symbols.extend(f"{module}.{symbol}" if module else symbol
                           for symbol in indexed.ast_record.qualified_symbols)
        vocabulary, weights, config = _configuration(symbols)
        if (vocabulary, weights, config) != (old_vocabulary, old_weights, old_config):
            raise RetrievalRefreshUnavailable("lexical_vocabulary_changed")
        scope = content_identity({"schema": "permitted-vector-inputs@1", "sources": hashes})
        snapshot = build_code_symbol_vector_index(evidence.ast_index,
            forest_id=scope, tree_id=scope, coverage_id=evidence.ast_index.index_id,
            included_paths=original.included_paths, excluded_paths=original.excluded_paths,
            producer_id=original.config.producer_id, chunker_id=original.config.chunker_id,
            normalization=original.config.normalization, model_id=original.config.model_id,
            model_revision=original.config.model_revision, dimensions=original.config.dimensions,
            metric=original.config.metric, configuration_id=config,
            vectors=lambda row: _vector(row.qualified_symbol, vocabulary, weights),
            previous=original, max_row_bytes=original.max_row_bytes)
        hits = search_code_symbol_vector_index(snapshot, _vector(query_text, vocabulary, weights),
            max_results=original_result.query.max_results)
        if retrieval._sources(repository, original.included_paths)[1] != hashes:
            raise ValueError("published lexical source changed during rebuild")
        return snapshot, hits
    return rebuild
