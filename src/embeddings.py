"""
embeddings.py
-------------
CLIP-based image and text embedding generation for the
AI E-Commerce Product Intelligence System.
"""

import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm


# ── device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(model_name: str = "ViT-B/32"):
    """Load CLIP model and preprocessing pipeline."""
    model, preprocess = clip.load(model_name, device=DEVICE)
    model.eval()
    return model, preprocess


def get_image_embedding(image_path: str, model, preprocess) -> np.ndarray:
    """
    Generate a normalised CLIP embedding for a single image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model : CLIP model
    preprocess : CLIP preprocess pipeline

    Returns
    -------
    np.ndarray  shape (512,)
    """
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)   # L2 normalise
    return embedding.cpu().numpy().astype("float32").flatten()


def get_text_embedding(text: str, model) -> np.ndarray:
    """
    Generate a normalised CLIP embedding for a text query.

    Parameters
    ----------
    text : str
        Natural-language product query.
    model : CLIP model

    Returns
    -------
    np.ndarray  shape (512,)
    """
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32").flatten()


def build_embedding_index(
    image_dir: str,
    model,
    preprocess,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a matrix of CLIP embeddings for every image in *image_dir*.

    Parameters
    ----------
    image_dir   : directory containing product images
    model       : CLIP model
    preprocess  : CLIP preprocess pipeline
    batch_size  : number of images processed per batch

    Returns
    -------
    embeddings  : np.ndarray  shape (N, 512)
    image_paths : list of absolute image paths in the same order
    """
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in '{image_dir}'")

    all_embeddings: list[np.ndarray] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
                valid_paths.append(p)
            except Exception as exc:
                print(f"  ⚠ Skipping {p}: {exc}")

        if not images:
            continue

        batch_tensor = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            embs = model.encode_image(batch_tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embeddings.append(embs.cpu().numpy().astype("float32"))

    embeddings = np.vstack(all_embeddings)
    return embeddings, image_paths


def save_embeddings(embeddings: np.ndarray, paths: list[str], save_dir: str = "data"):
    """Persist embeddings and path list to disk."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(save_dir, "image_paths.txt"), "w") as f:
        f.write("\n".join(paths))
    print(f"✅ Saved {len(paths)} embeddings to '{save_dir}/'")


def load_embeddings(save_dir: str = "data") -> tuple[np.ndarray, list[str]]:
    """Load pre-computed embeddings and path list from disk."""
    embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
    with open(os.path.join(save_dir, "image_paths.txt")) as f:
        paths = f.read().splitlines()
    return embeddings, paths
