"""
search.py
---------
FAISS-powered similarity search for text-to-product and image-to-product
queries in the AI E-Commerce Product Intelligence System.
"""

import os
import faiss
import numpy as np
import pandas as pd
from typing import Optional

from src.embeddings import (
    get_image_embedding,
    get_text_embedding,
    load_embeddings,
)


# ── FAISS index helpers ──────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product (cosine) index from an embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray  shape (N, D) — must already be L2-normalised

    Returns
    -------
    faiss.IndexFlatIP
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: str = "data/faiss.index"):
    """Persist the FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"✅ FAISS index saved to '{path}'")


def load_faiss_index(path: str = "data/faiss.index") -> faiss.IndexFlatIP:
    """Load a persisted FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at '{path}'. Run build_index.py first.")
    return faiss.read_index(path)


# ── search functions ──────────────────────────────────────────────────────────

def search_by_text(
    query: str,
    model,
    faiss_index: faiss.IndexFlatIP,
    image_paths: list[str],
    metadata: Optional[pd.DataFrame],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Natural-language product search.

    Parameters
    ----------
    query       : free-text query, e.g. "blue casual shirt"
    model       : CLIP model
    faiss_index : pre-built FAISS index
    image_paths : ordered list of image paths matching the FAISS index
    metadata    : optional DataFrame with columns id, productDisplayName, etc.
    top_k       : number of results to return

    Returns
    -------
    pd.DataFrame with columns: rank, image_path, score, [metadata cols]
    """
    query_emb = get_text_embedding(query, model).reshape(1, -1)
    scores, indices = faiss_index.search(query_emb, top_k)

    results = _build_result_df(indices[0], scores[0], image_paths, metadata)
    results.insert(0, "rank", range(1, len(results) + 1))
    return results


def search_by_image(
    image_path: str,
    model,
    preprocess,
    faiss_index: faiss.IndexFlatIP,
    image_paths: list[str],
    metadata: Optional[pd.DataFrame],
    top_k: int = 10,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Visual similarity search — given an image, find the most similar products.

    Parameters
    ----------
    image_path   : path to query image
    exclude_self : if True, skip the exact query image from results
    """
    query_emb = get_image_embedding(image_path, model, preprocess).reshape(1, -1)
    scores, indices = faiss_index.search(query_emb, top_k + (1 if exclude_self else 0))

    flat_indices = indices[0].tolist()
    flat_scores = scores[0].tolist()

    if exclude_self:
        # drop the result whose path matches the query
        filtered = [
            (idx, sc)
            for idx, sc in zip(flat_indices, flat_scores)
            if image_paths[idx] != image_path
        ][:top_k]
        flat_indices, flat_scores = zip(*filtered) if filtered else ([], [])

    results = _build_result_df(list(flat_indices), list(flat_scores), image_paths, metadata)
    results.insert(0, "rank", range(1, len(results) + 1))
    return results


def _build_result_df(
    indices: list[int],
    scores: list[float],
    image_paths: list[str],
    metadata: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Internal helper — combine FAISS results with metadata."""
    rows = []
    for idx, score in zip(indices, scores):
        row = {"image_path": image_paths[idx], "score": round(float(score), 4)}
        if metadata is not None:
            img_id = os.path.splitext(os.path.basename(image_paths[idx]))[0]
            try:
                img_id_int = int(img_id)
                meta_row = metadata[metadata["id"] == img_id_int]
                if not meta_row.empty:
                    row.update(meta_row.iloc[0].to_dict())
            except ValueError:
                pass
        rows.append(row)
    return pd.DataFrame(rows)
