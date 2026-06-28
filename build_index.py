"""
build_index.py
--------------
One-time script: encode all product images with CLIP and build the FAISS index.

Usage
-----
    python build_index.py --image_dir images/ --data_dir data/

This must be run before launching the Streamlit app.
"""

import argparse
import faiss

from src.embeddings import load_clip_model, build_embedding_index, save_embeddings
from src.search import build_faiss_index, save_faiss_index
from src.utils import ensure_dirs


def main(image_dir: str, data_dir: str, model_name: str):
    ensure_dirs(data_dir)

    print(f"📦 Loading CLIP model '{model_name}' …")
    model, preprocess = load_clip_model(model_name)

    print(f"🖼  Encoding images in '{image_dir}' …")
    embeddings, image_paths = build_embedding_index(image_dir, model, preprocess)

    print("💾 Saving embeddings …")
    save_embeddings(embeddings, image_paths, data_dir)

    print("🔍 Building FAISS index …")
    index = build_faiss_index(embeddings)

    print("💾 Saving FAISS index …")
    save_faiss_index(index, path=f"{data_dir}/faiss.index")

    print(
        f"\n✅ Done — indexed {len(image_paths)} products.\n"
        f"   Run  'streamlit run app.py'  to launch the web app."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP+FAISS product index")
    parser.add_argument("--image_dir", default="images/", help="Directory of product images")
    parser.add_argument("--data_dir",  default="data/",   help="Output directory for index files")
    parser.add_argument("--model",     default="ViT-B/32", help="CLIP model variant")
    args = parser.parse_args()

    main(args.image_dir, args.data_dir, args.model)
