#!/usr/bin/env bash
# =============================================================================
# setup_and_deploy.sh
# Step-by-step guide to set up, run locally, and push to GitHub.
# Run individual sections — don't execute the whole script at once.
# =============================================================================

set -e   # exit on any error

echo ""
echo "=============================================="
echo "  AI E-Commerce Product Intelligence System"
echo "  Setup & Deployment Guide"
echo "=============================================="
echo ""

# ── STEP 1: Create virtual environment ────────────────────────────────────────
echo "STEP 1: Creating Python virtual environment …"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
echo "✅ Virtual environment created"

# ── STEP 2: Install dependencies ──────────────────────────────────────────────
echo ""
echo "STEP 2: Installing dependencies …"
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# ── STEP 3: Download dataset (manual step) ────────────────────────────────────
echo ""
echo "STEP 3: Dataset download (manual)"
echo "  1. Go to: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset"
echo "  2. Download and unzip"
echo "  3. Copy 'styles.csv' → data/styles.csv"
echo "  4. Copy 'images/' folder → images/"
read -p "  Press Enter once dataset is ready …"

# ── STEP 4: Build CLIP + FAISS index ──────────────────────────────────────────
echo ""
echo "STEP 4: Building CLIP + FAISS index …"
echo "  This takes ~10-20 min on CPU, ~2 min on GPU."
python build_index.py --image_dir images/ --data_dir data/
echo "✅ Index built"

# ── STEP 5: Run Streamlit app locally ─────────────────────────────────────────
echo ""
echo "STEP 5: Launching Streamlit app …"
echo "  Open http://localhost:8501 in your browser"
streamlit run app.py &

# ── STEP 6: Git init and push to GitHub ───────────────────────────────────────
# (run AFTER testing locally)
echo ""
echo "STEP 6: Pushing to GitHub"
echo "  Replace <your-username> below with your GitHub username"
echo ""

git init
git add .
git commit -m "feat: initial commit — AI E-Commerce Product Intelligence System

- CLIP + FAISS multimodal search (text and image)
- Complementary product recommendations
- Duplicate catalogue detection
- Zero-shot AI classification (CLIP + ResNet50/EfficientNet)
- Streamlit web app with 5 feature panels
- Modular src/ package with full docstrings"

git branch -M main

# Change this to your actual repo URL:
GITHUB_USERNAME="Triveni1706"
REPO_NAME="AI-Ecommerce-Product-Intelligence"
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
git push -u origin main

echo ""
echo "✅ Project live at: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""

# ── STEP 7: Deploy to Streamlit Community Cloud (free) ────────────────────────
echo "STEP 7: Streamlit Cloud Deployment (free hosting)"
echo ""
echo "  1. Go to https://streamlit.io/cloud and sign in with GitHub"
echo "  2. Click 'New app'"
echo "  3. Repository: ${GITHUB_USERNAME}/${REPO_NAME}"
echo "  4. Branch: main"
echo "  5. Main file path: app.py"
echo "  6. Click Deploy"
echo ""
echo "  ⚠ Note: Streamlit Cloud has 1GB RAM — for full dataset you may need"
echo "    to add the FAISS index and embeddings.npy via Streamlit secrets or"
echo "    host them on Google Drive / Hugging Face Hub and download at startup."
echo ""
echo "  Alternative: Deploy on Render.com (free tier)"
echo "    1. Create a Web Service on render.com"
echo "    2. Set Build command: pip install -r requirements.txt"
echo "    3. Set Start command: streamlit run app.py --server.port=\$PORT --server.headless=true"
echo ""
echo "=============================================="
echo "  All done! 🎉"
echo "=============================================="
