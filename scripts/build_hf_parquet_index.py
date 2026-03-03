#!/usr/bin/env python3
"""Download HF parquet files and build a vector index using embeddings_router."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ipfs_accelerate_py.embeddings_router import embed_texts
from ipfs_accelerate_py.embeddings.ipfs_knn_index import IPFSKnnIndex


def _run(cmd: List[str], env: Optional[dict] = None) -> None:
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _pick_text_column(schema: pa.Schema) -> str:
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            return field.name
    raise ValueError(
        "No string-like column found. Pass --text-column to select a column manually."
    )


def _iter_parquet_texts(
    parquet_path: Path,
    text_column: Optional[str],
    batch_size: int,
    metadata_columns: Iterable[str],
) -> Iterable[tuple[List[str], List[dict]]]:
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow

    text_col = text_column or _pick_text_column(schema)
    if text_col not in schema.names:
        raise ValueError(f"Text column '{text_col}' not found in {parquet_path}")

    requested_cols = [text_col] + [col for col in metadata_columns if col in schema.names]
    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=requested_cols):
        batch_table = pa.Table.from_batches([batch])
        texts = batch_table[text_col].to_pylist()

        rows_meta = []
        for idx in range(len(texts)):
            meta = {"row_index": row_offset + idx}
            for col in metadata_columns:
                if col in batch_table.column_names:
                    meta[col] = batch_table[col][idx].as_py()
            rows_meta.append(meta)

        yield texts, rows_meta
        row_offset += len(texts)


def build_index(args: argparse.Namespace) -> None:
    download_dir = Path(args.download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        cmd = [
            "huggingface-cli",
            "download",
            args.repo_id,
            "--repo-type",
            "dataset",
            "--include",
            f"{args.subdir}/*.parquet",
            "--local-dir",
            str(download_dir),
            "--local-dir-use-symlinks",
            "False",
        ]
        env = os.environ.copy()
        if args.hf_token_env and os.getenv(args.hf_token_env):
            env["HUGGINGFACE_HUB_TOKEN"] = os.getenv(args.hf_token_env, "")
        _run(cmd, env=env)

    parquet_root = download_dir / args.subdir
    parquet_files = sorted(parquet_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {parquet_root}")

    model_name = args.model_name or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL") or "thenlper/gte-small"
    provider = args.provider or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER")

    index: Optional[IPFSKnnIndex] = None
    total_rows = 0

    for parquet_path in parquet_files:
        for texts, meta in _iter_parquet_texts(
            parquet_path,
            args.text_column,
            args.batch_size,
            args.metadata_columns,
        ):
            filtered = []
            filtered_meta = []
            for text, row_meta in zip(texts, meta):
                if text is None:
                    continue
                text_str = str(text).strip()
                if not text_str:
                    continue
                row_meta["source_file"] = str(parquet_path)
                row_meta["text"] = text_str
                filtered.append(text_str)
                filtered_meta.append(row_meta)

            if not filtered:
                continue

            vectors = embed_texts(
                filtered,
                model_name=model_name,
                provider=provider,
            )
            vectors_np = np.asarray(vectors, dtype=np.float32)

            if index is None:
                index = IPFSKnnIndex(dimension=vectors_np.shape[1], metric=args.metric)

            index.add_vectors(vectors_np, metadata=filtered_meta)
            total_rows += len(filtered)

            if args.max_rows and total_rows >= args.max_rows:
                break

        if args.max_rows and total_rows >= args.max_rows:
            break

    if index is None:
        raise RuntimeError("No rows were indexed. Check text column selection or max_rows.")

    print(f"Indexed {total_rows} rows with dimension {index.dimension}")

    if args.save_to_ipfs:
        cid = index.save_to_ipfs()
        print(f"Index saved to IPFS with CID: {cid}")
        if args.output_cid_file:
            Path(args.output_cid_file).write_text(cid, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="justicedao/ipfs_state_laws",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--subdir",
        default="OR/parsed/parquet",
        help="Subdirectory with parquet files inside the repo",
    )
    parser.add_argument(
        "--download-dir",
        default="data/hf_parquet",
        help="Local directory for HF download",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and use local parquet files",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Column name to embed; defaults to the first string column",
    )
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=[],
        help="Additional columns to store as metadata",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows to index",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Embedding model name (default: env or thenlper/gte-small)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Embeddings provider (default: router selection)",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean", "dot"],
        help="Similarity metric",
    )
    parser.add_argument(
        "--save-to-ipfs",
        action="store_true",
        help="Persist the index to IPFS and print the CID",
    )
    parser.add_argument(
        "--output-cid-file",
        default=None,
        help="File to write the CID to (when --save-to-ipfs is set)",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Env var name holding the HF token for CLI (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
