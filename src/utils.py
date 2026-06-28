"""
utils.py
--------
Shared utility functions for the AI E-Commerce Product Intelligence System.
"""

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ── metadata ──────────────────────────────────────────────────────────────────

def load_metadata(csv_path: str = "data/styles.csv") -> pd.DataFrame:
    """
    Load the Kaggle Fashion Product Dataset metadata CSV.

    Expected columns (subset used):
        id, gender, masterCategory, subCategory, articleType,
        baseColour, season, year, usage, productDisplayName
    """
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").dropna().astype(int)
    # clean up whitespace in string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df


def get_product_info(product_id: int, metadata: pd.DataFrame) -> dict:
    """Return a dict of metadata for a given product id."""
    row = metadata[metadata["id"] == product_id]
    if row.empty:
        return {"id": product_id}
    return row.iloc[0].to_dict()


def id_to_image_path(
    product_id: int,
    image_dir: str = "images",
    ext: str = ".jpg",
) -> str | None:
    """Resolve image file path from a product id."""
    path = os.path.join(image_dir, f"{product_id}{ext}")
    return path if os.path.exists(path) else None


# ── image helpers ─────────────────────────────────────────────────────────────

def load_image(path: str, max_size: int = 512) -> Image.Image:
    """Load and optionally resize an image for display."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def make_image_grid(
    image_paths: list[str],
    cols: int = 4,
    cell_size: tuple[int, int] = (150, 200),
) -> Image.Image:
    """
    Tile a list of product images into a single grid image.

    Parameters
    ----------
    image_paths : ordered list of image paths
    cols        : number of columns in the grid
    cell_size   : (width, height) of each cell in pixels

    Returns
    -------
    PIL.Image.Image
    """
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    w, h = cell_size
    grid = Image.new("RGB", (cols * w, rows * h), color=(245, 245, 245))

    for i, path in enumerate(image_paths):
        try:
            img = load_image(path, max_size=max(w, h))
            img = img.resize(cell_size, Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", cell_size, (220, 220, 220))

        row, col = divmod(i, cols)
        grid.paste(img, (col * w, row * h))

    return grid


# ── result formatting ─────────────────────────────────────────────────────────

def format_results_table(df: pd.DataFrame, max_name_len: int = 40) -> str:
    """Return a pretty-printed text table of search / recommendation results."""
    display_cols = ["rank", "score", "productDisplayName", "subCategory", "baseColour", "image_path"]
    display_cols = [c for c in display_cols if c in df.columns]

    lines = []
    header = " | ".join(f"{c:<{max_name_len if c == 'productDisplayName' else 15}}" for c in display_cols)
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in df.iterrows():
        parts = []
        for c in display_cols:
            val = str(row.get(c, ""))
            width = max_name_len if c == "productDisplayName" else 15
            parts.append(f"{val[:width]:<{width}}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def print_duplicate_report(groups: list[dict]) -> None:
    """Pretty-print a duplicate group report to stdout."""
    print(f"\n{'='*60}")
    print(f"  DUPLICATE DETECTION REPORT — {len(groups)} groups found")
    print(f"{'='*60}\n")
    for g in groups[:20]:   # show first 20 groups
        print(f"Group {g['group_id']:03d} — {g['size']} similar products:")
        for m in g["members"]:
            score_str = f"score={m['score']:.3f}"
            name = m.get("name", os.path.basename(m["image_path"]))
            print(f"  {score_str}  {name}")
        print()


# ── file helpers ──────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def list_images(directory: str) -> list[str]:
    """Return sorted list of image paths in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if Path(f).suffix.lower() in exts
    )
