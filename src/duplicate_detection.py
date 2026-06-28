"""
duplicate_detection.py
----------------------
Detect visually similar / duplicate product listings in an e-commerce catalogue
using CLIP embeddings and FAISS similarity search.

Two products are considered duplicates when their cosine similarity exceeds
a configurable threshold (default 0.97).
"""

import os
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── core detection ────────────────────────────────────────────────────────────

def find_duplicates(
    embeddings: np.ndarray,
    image_paths: list[str],
    metadata: pd.DataFrame | None = None,
    threshold: float = 0.97,
    batch_size: int = 128,
) -> list[dict]:
    """
    Identify groups of duplicate / near-duplicate product images.

    Parameters
    ----------
    embeddings  : np.ndarray  shape (N, D) — L2-normalised CLIP embeddings
    image_paths : list of image paths matching the embedding matrix rows
    metadata    : optional DataFrame with product metadata (id, name, …)
    threshold   : cosine-similarity cutoff — pairs above this are duplicates
    batch_size  : images processed per FAISS query batch

    Returns
    -------
    list of dicts, each describing one duplicate group:
        {
            "group_id": int,
            "size": int,
            "members": [{"image_path": str, "score": float, "id": int?, "name": str?}]
        }
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    n = len(image_paths)
    visited = set()
    groups: list[dict] = []
    group_id = 0

    for i in tqdm(range(0, n, batch_size), desc="Scanning for duplicates"):
        batch_embs = embeddings[i : i + batch_size]
        scores_batch, indices_batch = index.search(batch_embs, 20)   # top-20 per query

        for local_j, (scores, neighbours) in enumerate(
            zip(scores_batch, indices_batch)
        ):
            global_i = i + local_j
            if global_i in visited:
                continue

            # collect neighbours above threshold (exclude self)
            dupes = [
                (int(n_idx), float(sc))
                for n_idx, sc in zip(neighbours, scores)
                if n_idx != global_i and sc >= threshold
            ]

            if not dupes:
                continue

            # mark all members as visited so we don't create duplicate groups
            all_members_idx = [global_i] + [d[0] for d in dupes]
            if all(idx in visited for idx in all_members_idx):
                continue

            for idx in all_members_idx:
                visited.add(idx)

            members = _build_members(
                [(global_i, 1.0)] + dupes, image_paths, metadata
            )
            groups.append(
                {"group_id": group_id, "size": len(members), "members": members}
            )
            group_id += 1

    groups.sort(key=lambda g: g["size"], reverse=True)
    print(
        f"✅ Duplicate scan complete — {len(groups)} duplicate groups found "
        f"({sum(g['size'] for g in groups)} total affected products)"
    )
    return groups


def duplicates_to_dataframe(groups: list[dict]) -> pd.DataFrame:
    """Flatten duplicate groups into a tidy DataFrame for display / export."""
    rows = []
    for g in groups:
        for m in g["members"]:
            rows.append({"group_id": g["group_id"], "group_size": g["size"], **m})
    return pd.DataFrame(rows)


def suggest_deduplication(groups: list[dict]) -> pd.DataFrame:
    """
    For each duplicate group, recommend which product to KEEP (highest score /
    primary listing) and which to REMOVE.

    Returns a DataFrame with columns: group_id, action, image_path, product_name
    """
    rows = []
    for g in groups:
        members = g["members"]
        keep = members[0]   # first member is the query; others are dupes
        rows.append(
            {
                "group_id": g["group_id"],
                "action": "KEEP",
                "image_path": keep["image_path"],
                "product_name": keep.get("name", "—"),
                "score": keep["score"],
            }
        )
        for m in members[1:]:
            rows.append(
                {
                    "group_id": g["group_id"],
                    "action": "REMOVE",
                    "image_path": m["image_path"],
                    "product_name": m.get("name", "—"),
                    "score": m["score"],
                }
            )
    return pd.DataFrame(rows)


# ── statistics ────────────────────────────────────────────────────────────────

def duplicate_statistics(groups: list[dict], total_products: int) -> dict:
    """Return a summary statistics dict for the duplicate report."""
    total_dupes = sum(g["size"] for g in groups)
    removable = sum(g["size"] - 1 for g in groups)   # keep 1 per group
    return {
        "total_products": total_products,
        "duplicate_groups": len(groups),
        "affected_products": total_dupes,
        "removable_listings": removable,
        "catalogue_bloat_pct": round(removable / total_products * 100, 2),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _build_members(
    idx_score_pairs: list[tuple[int, float]],
    image_paths: list[str],
    metadata: pd.DataFrame | None,
) -> list[dict]:
    members = []
    for idx, score in idx_score_pairs:
        m: dict = {
            "image_path": image_paths[idx],
            "score": round(score, 4),
        }
        if metadata is not None:
            pid = _path_to_id(image_paths[idx])
            meta_row = metadata[metadata["id"] == pid]
            if not meta_row.empty:
                m["id"] = pid
                m["name"] = meta_row.iloc[0].get("productDisplayName", "")
                m["sub_category"] = meta_row.iloc[0].get("subCategory", "")
        members.append(m)
    return members


def _path_to_id(path: str) -> int:
    try:
        return int(os.path.splitext(os.path.basename(path))[0])
    except ValueError:
        return -1
