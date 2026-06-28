"""
app.py
------
AI-Powered E-Commerce Product Intelligence System
Streamlit web application — run with:  streamlit run app.py
"""

import os
import io
import sys
import textwrap

import numpy as np
import streamlit as st
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embeddings import load_clip_model, get_text_embedding, get_image_embedding, load_embeddings
from search import load_faiss_index, search_by_text, search_by_image, build_faiss_index
from recommendation import recommend_complementary, recommend_similar
from duplicate_detection import find_duplicates, duplicate_statistics, suggest_deduplication
from classification import clip_zero_shot_classify, format_prediction
from utils import load_metadata, load_image, make_image_grid, list_images

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI E-Commerce Product Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .score-badge {
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .category-badge {
        background: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.9rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── session-state cached resources ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model …")
def _load_model():
    return load_clip_model("ViT-B/32")


@st.cache_resource(show_spinner="Loading FAISS index …")
def _load_index():
    return load_faiss_index("data/faiss.index")


@st.cache_resource(show_spinner="Loading embeddings …")
def _load_embs():
    return load_embeddings("data/")


@st.cache_resource(show_spinner="Loading product metadata …")
def _load_meta():
    if os.path.exists("data/styles.csv"):
        return load_metadata("data/styles.csv")
    return None


def index_ready() -> bool:
    return (
        os.path.exists("data/faiss.index")
        and os.path.exists("data/embeddings.npy")
        and os.path.exists("data/image_paths.txt")
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-bag.png", width=64)
    st.markdown("## 🛍️ Product Intelligence")
    st.markdown("---")

    feature = st.radio(
        "Choose a feature",
        [
            "🏠  Overview",
            "🔍  Text Search",
            "🖼️  Image Search",
            "💡  Recommendations",
            "🔁  Duplicate Detection",
            "🏷️  AI Classification",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Kaggle Fashion Product Images Dataset")

    st.markdown("**Model**")
    st.caption("OpenAI CLIP ViT-B/32 + FAISS")

    if not index_ready():
        st.warning("⚠️ Index not built yet.\n\nRun:\n```\npython build_index.py\n```")
    else:
        st.success("✅ Index ready")


# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">AI E-Commerce Product Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CLIP · FAISS · Multimodal Search · Recommendations · Duplicate Detection · AI Classification</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if "Overview" in feature:
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("🔍", "Text Search", "Natural language → Products"),
        ("🖼️", "Image Search", "Photo → Similar Products"),
        ("💡", "Recommendations", "Product → Complements"),
        ("🔁", "Deduplication", "Find Duplicate Listings"),
        ("🏷️", "Classification", "Auto-Label Products"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="text-align:center">
                <div style="font-size:2rem">{icon}</div>
                <strong>{title}</strong>
                <p style="color:#6b7280;font-size:0.8rem;margin-top:4px">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Step 1 — CLIP Embedding**
Every product image is passed through OpenAI's CLIP vision encoder,
producing a 512-dimensional semantic vector that captures both visual
appearance and conceptual meaning.

**Step 2 — FAISS Indexing**
All embeddings are stored in a FAISS inner-product index enabling
sub-millisecond similarity search over hundreds of thousands of products.
        """)
    with c2:
        st.markdown("""
**Step 3 — Multimodal Query**
Both image and text queries are embedded in the same CLIP space, so
*"blue running shoes"* and a photo of blue shoes retrieve the same results.

**Step 4 — Downstream Intelligence**
Similarity scores power recommendations, duplicate clustering,
and zero-shot classification — all without any labelled training data.
        """)

    st.info("👈 Select a feature from the sidebar to get started.")


# ════════════════════════════════════════════════════════════════════════════════
# TEXT SEARCH
# ════════════════════════════════════════════════════════════════════════════════
elif "Text Search" in feature:
    st.header("🔍 Natural Language Product Search")
    st.caption("Describe what you're looking for in plain English.")

    query = st.text_input(
        "Search query",
        placeholder="e.g. Blue casual shirt, Black running shoes, Women red dress …",
    )
    top_k = st.slider("Number of results", 4, 20, 8)

    if st.button("Search", type="primary") and query:
        if not index_ready():
            st.error("Please build the index first: `python build_index.py`")
        else:
            with st.spinner("Searching …"):
                model, preprocess = _load_model()
                faiss_index = _load_index()
                _, image_paths = _load_embs()
                metadata = _load_meta()

                results = search_by_text(query, model, faiss_index, image_paths, metadata, top_k)

            st.success(f"Found {len(results)} results for **'{query}'**")
            cols = st.columns(4)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 4]:
                    try:
                        img = load_image(row["image_path"])
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.image("https://via.placeholder.com/150x200?text=No+Image")
                    name = row.get("productDisplayName", os.path.basename(row["image_path"]))
                    st.caption(f"**{str(name)[:40]}**")
                    score = row.get("score", 0)
                    st.markdown(f'<span class="score-badge">Score: {score:.3f}</span>', unsafe_allow_html=True)
                    sub = row.get("subCategory", "")
                    if sub:
                        st.caption(f"📦 {sub}")


# ════════════════════════════════════════════════════════════════════════════════
# IMAGE SEARCH
# ════════════════════════════════════════════════════════════════════════════════
elif "Image Search" in feature:
    st.header("🖼️ Visual Similarity Search")
    st.caption("Upload a product image to find visually similar items.")

    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
    top_k = st.slider("Number of results", 4, 20, 8)

    if uploaded and st.button("Find Similar Products", type="primary"):
        if not index_ready():
            st.error("Please build the index first: `python build_index.py`")
        else:
            # save temp file
            tmp_path = "data/_query_temp.jpg"
            Image.open(uploaded).convert("RGB").save(tmp_path)

            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(tmp_path, caption="Your query image", use_container_width=True)

            with st.spinner("Searching …"):
                model, preprocess = _load_model()
                faiss_index = _load_index()
                _, image_paths = _load_embs()
                metadata = _load_meta()

                results = search_by_image(
                    tmp_path, model, preprocess, faiss_index, image_paths, metadata, top_k
                )

            with c2:
                st.success(f"Top {len(results)} similar products")
                cols = st.columns(4)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 4]:
                        try:
                            img = load_image(row["image_path"])
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.image("https://via.placeholder.com/150x200?text=No+Image")
                        st.caption(f"Score: **{row.get('score', 0):.3f}**")
                        name = row.get("productDisplayName", "")
                        if name:
                            st.caption(str(name)[:35])


# ════════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════════
elif "Recommendations" in feature:
    st.header("💡 Complementary Product Recommendations")
    st.caption("Upload a product image to discover what pairs well with it.")

    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
    rec_type = st.radio("Recommendation type", ["Complementary (cross-category)", "Similar (same category)"])

    if uploaded and st.button("Get Recommendations", type="primary"):
        if not index_ready():
            st.error("Please build the index first.")
        else:
            tmp_path = "data/_rec_temp.jpg"
            Image.open(uploaded).convert("RGB").save(tmp_path)

            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(tmp_path, caption="Query product", use_container_width=True)

            with st.spinner("Generating recommendations …"):
                model, preprocess = _load_model()
                faiss_index = _load_index()
                _, image_paths = _load_embs()
                metadata = _load_meta()

                if "Complementary" in rec_type and metadata is not None:
                    recs = recommend_complementary(
                        tmp_path, model, preprocess, faiss_index, image_paths, metadata
                    )
                    with c2:
                        for cat, df_cat in recs.items():
                            st.markdown(f"#### {cat}")
                            cols = st.columns(min(4, len(df_cat)))
                            for i, (_, row) in enumerate(df_cat.iterrows()):
                                with cols[i % len(cols)]:
                                    try:
                                        img = load_image(row["image_path"])
                                        st.image(img, use_container_width=True)
                                    except Exception:
                                        pass
                                    st.caption(f"Score: {row.get('score', 0):.3f}")
                else:
                    results = recommend_similar(
                        tmp_path, model, preprocess, faiss_index, image_paths, metadata
                    )
                    with c2:
                        cols = st.columns(4)
                        for i, (_, row) in enumerate(results.iterrows()):
                            with cols[i % 4]:
                                try:
                                    img = load_image(row["image_path"])
                                    st.image(img, use_container_width=True)
                                except Exception:
                                    pass
                                st.caption(f"Score: {row.get('score', 0):.3f}")


# ════════════════════════════════════════════════════════════════════════════════
# DUPLICATE DETECTION
# ════════════════════════════════════════════════════════════════════════════════
elif "Duplicate" in feature:
    st.header("🔁 Duplicate Product Detection")
    st.caption("Scan the catalogue to find duplicate or near-duplicate product listings.")

    threshold = st.slider("Similarity threshold", 0.90, 0.99, 0.97, step=0.01,
                          help="Products with cosine similarity ≥ threshold are flagged as duplicates.")

    if st.button("Scan for Duplicates", type="primary"):
        if not index_ready():
            st.error("Please build the index first.")
        else:
            with st.spinner("Scanning catalogue …"):
                embeddings, image_paths = _load_embs()
                metadata = _load_meta()
                groups = find_duplicates(embeddings, image_paths, metadata, threshold=threshold)
                stats = duplicate_statistics(groups, len(image_paths))

            # stats row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Products", stats["total_products"])
            c2.metric("Duplicate Groups", stats["duplicate_groups"])
            c3.metric("Affected Products", stats["affected_products"])
            c4.metric("Catalogue Bloat", f"{stats['catalogue_bloat_pct']}%")

            if groups:
                st.markdown("---")
                st.markdown("### Duplicate Groups (top 10)")
                for g in groups[:10]:
                    with st.expander(f"Group {g['group_id']} — {g['size']} similar products"):
                        cols = st.columns(min(g["size"], 5))
                        for i, m in enumerate(g["members"][:5]):
                            with cols[i]:
                                try:
                                    img = load_image(m["image_path"])
                                    st.image(img, use_container_width=True)
                                except Exception:
                                    pass
                                action = "✅ KEEP" if i == 0 else "❌ REMOVE"
                                st.caption(f"{action}  (sim={m['score']:.3f})")
                                if "name" in m:
                                    st.caption(str(m["name"])[:30])
            else:
                st.success("✅ No duplicates found above the chosen threshold!")


# ════════════════════════════════════════════════════════════════════════════════
# AI CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════
elif "Classification" in feature:
    st.header("🏷️ AI Product Classification")
    st.caption("Automatically predict the hierarchical category of any product image.")

    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded and st.button("Classify Product", type="primary"):
        tmp_path = "data/_cls_temp.jpg"
        Image.open(uploaded).convert("RGB").save(tmp_path)

        with st.spinner("Classifying …"):
            model, preprocess = _load_model()
            pred = clip_zero_shot_classify(tmp_path, model, preprocess)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(tmp_path, caption="Input image", use_container_width=True)

        with c2:
            st.markdown("#### Predicted Category")
            label = format_prediction(pred)
            st.markdown(f'<span class="category-badge">{label}</span>', unsafe_allow_html=True)
            st.markdown("")

            st.markdown("#### Confidence per level")
            conf = pred.get("confidence", {})

            for level in ["gender", "masterCategory", "subCategory"]:
                val = pred.get(level, "?")
                score = conf.get(level, 0)
                st.markdown(f"**{level.capitalize()}:** {val}")
                st.progress(float(score), text=f"{score*100:.1f}%")
                st.markdown("")

            st.info(
                "ℹ️ Classification uses CLIP zero-shot inference — no labelled "
                "training data required. Accuracy improves further with a "
                "fine-tuned ResNet50/EfficientNet head (see `src/classification.py`)."
            )
