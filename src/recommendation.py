"""
recommendation.py
-----------------
Complementary product recommendation engine for the
AI E-Commerce Product Intelligence System.

Strategy
--------
1. Retrieve the top-K visually similar items to the query product.
2. Filter results to *different* sub-categories (complementary logic).
3. Optionally boost results from a complementary-category allowlist.

Example complementary map
    Shoes   → Socks, Watches, Bags, Sunglasses
    Shirts  → Trousers, Belts, Watches
    Dresses → Handbags, Heels, Earrings
"""

import os
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from typing import Optional

from embeddings import get_image_embedding, get_text_embedding


# ── complementary category map ────────────────────────────────────────────────
# Keys are subCategory values in the Kaggle Fashion dataset.
COMPLEMENTARY_MAP: dict[str, list[str]] = {
    "Shoes": ["Socks", "Watches", "Bags", "Sunglasses", "Belts"],
    "Casual Shoes": ["Socks", "Watches", "Backpacks", "Caps"],
    "Sports Shoes": ["Socks", "Sports Accessories", "Caps", "Shorts"],
    "Formal Shoes": ["Belts", "Watches", "Ties", "Socks"],
    "Sandals": ["Sunglasses", "Bags", "Jewellery"],
    "Tops": ["Trousers", "Jeans", "Skirts", "Belts"],
    "Shirts": ["Trousers", "Belts", "Watches", "Ties"],
    "Tshirts": ["Jeans", "Track Pants", "Caps", "Backpacks"],
    "Jeans": ["Shirts", "Tshirts", "Belts", "Shoes"],
    "Dresses": ["Handbags", "Heels", "Jewellery", "Sunglasses"],
    "Watches": ["Shirts", "Formal Shoes", "Belts"],
    "Bags": ["Shoes", "Wallets", "Sunglasses"],
    "Sunglasses": ["Caps", "Shoes", "Bags"],
}

DEFAULT_FALLBACK_CATEGORIES = ["Watches", "Bags", "Sunglasses", "Belts", "Socks"]


def get_complementary_categories(sub_category: str) -> list[str]:
    """Return a list of sub-categories that complement *sub_category*."""
    return COMPLEMENTARY_MAP.get(sub_category, DEFAULT_FALLBACK_CATEGORIES)


def recommend_complementary(
    query_image_path: str,
    model,
    preprocess,
    faiss_index: faiss.IndexFlatIP,
    image_paths: list[str],
    metadata: pd.DataFrame,
    top_k: int = 8,
    num_categories: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Given a query product image, recommend complementary products grouped
    by sub-category.

    Parameters
    ----------
    query_image_path : path to the query product image
    model            : CLIP model
    preprocess       : CLIP preprocessing pipeline
    faiss_index      : FAISS index over the product catalogue
    image_paths      : ordered list of paths corresponding to the index
    metadata         : DataFrame with columns id, subCategory, productDisplayName, …
    top_k            : total similar items to retrieve from FAISS
    num_categories   : how many complementary category groups to return

    Returns
    -------
    dict mapping sub_category_name → pd.DataFrame of recommended products
    """
    # 1. Embed the query image
    query_emb = get_image_embedding(query_image_path, model, preprocess).reshape(1, -1)

    # 2. Retrieve the query product's own sub-category from metadata
    img_id = _path_to_id(query_image_path)
    query_meta = metadata[metadata["id"] == img_id]
    query_sub_cat = query_meta["subCategory"].values[0] if not query_meta.empty else ""

    # 3. Find complementary sub-categories
    comp_cats = get_complementary_categories(query_sub_cat)[:num_categories]

    # 4. Large FAISS search to get a pool of candidates
    pool_size = min(200, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_emb, pool_size)

    # 5. Filter pool by complementary categories
    results: dict[str, pd.DataFrame] = {}
    for cat in comp_cats:
        cat_rows = []
        for idx, score in zip(indices[0], scores[0]):
            pid = _path_to_id(image_paths[idx])
            meta_row = metadata[metadata["id"] == pid]
            if meta_row.empty:
                continue
            row_cat = meta_row["subCategory"].values[0]
            if row_cat == cat:
                entry = meta_row.iloc[0].to_dict()
                entry["score"] = round(float(score), 4)
                entry["image_path"] = image_paths[idx]
                cat_rows.append(entry)
            if len(cat_rows) >= (top_k // num_categories):
                break
        if cat_rows:
            results[cat] = pd.DataFrame(cat_rows)

    return results


def recommend_similar(
    query_image_path: str,
    model,
    preprocess,
    faiss_index: faiss.IndexFlatIP,
    image_paths: list[str],
    metadata: Optional[pd.DataFrame],
    top_k: int = 6,
) -> pd.DataFrame:
    """
    Return *top_k* visually similar products in the same category (classic
    'customers also viewed' style).
    """
    query_emb = get_image_embedding(query_image_path, model, preprocess).reshape(1, -1)
    scores, indices = faiss_index.search(query_emb, top_k + 1)   # +1 to skip self

    rows = []
    for idx, score in zip(indices[0], scores[0]):
        path = image_paths[idx]
        if path == query_image_path:
            continue
        row: dict = {"image_path": path, "score": round(float(score), 4)}
        if metadata is not None:
            pid = _path_to_id(path)
            meta_row = metadata[metadata["id"] == pid]
            if not meta_row.empty:
                row.update(meta_row.iloc[0].to_dict())
        rows.append(row)
        if len(rows) == top_k:
            break

    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ── internal helpers ──────────────────────────────────────────────────────────

def _path_to_id(path: str) -> int:
    """Extract integer product id from image filename."""
    try:
        return int(os.path.splitext(os.path.basename(path))[0])
    except ValueError:
        return -1
