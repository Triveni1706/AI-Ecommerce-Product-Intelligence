"""
classification.py
-----------------
AI Product Classification — predict hierarchical category labels
(gender → masterCategory → subCategory) from a product image.

Two strategies available:
1. CLIP zero-shot  — no training required; uses text prompts as class prototypes
2. ResNet50 / EfficientNet fine-tuned — train a lightweight head on top of a
   frozen ImageNet backbone using the catalogue's own labels

Both produce a prediction in the form:
    {"gender": "Men", "masterCategory": "Footwear", "subCategory": "Running Shoes"}
"""

import os
import numpy as np
import torch
import torch.nn as nn
import clip
from PIL import Image
from torchvision import transforms, models


# ── device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── CLIP zero-shot classification ─────────────────────────────────────────────

# Hierarchical label taxonomy (subset of the Kaggle Fashion Product dataset)
TAXONOMY = {
    "gender": ["Men", "Women", "Boys", "Girls", "Unisex"],
    "masterCategory": [
        "Apparel", "Footwear", "Accessories", "Personal Care", "Sporting Goods",
    ],
    "subCategory": [
        "Topwear", "Bottomwear", "Watches", "Socks", "Shoes", "Belts",
        "Wallets", "Bags", "Sunglasses", "Makeup", "Headwear", "Jewellery",
        "Eyewear", "Mufflers", "Caps", "Sandals", "Flip Flops", "Innerwear",
        "Dress", "Sarees", "Formal Shoes", "Casual Shoes", "Sports Shoes",
        "Running Shoes", "Shorts", "Jeans", "Track Pants", "Tshirts", "Shirts",
        "Dresses", "Tops", "Kurtas", "Kurtis", "Salwar", "Churidar",
    ],
}


def clip_zero_shot_classify(
    image_path: str,
    model,
    preprocess,
    taxonomy: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """
    Classify a product image into gender / masterCategory / subCategory
    using CLIP zero-shot inference.

    Parameters
    ----------
    image_path : path to product image
    model      : loaded CLIP model
    preprocess : CLIP preprocess pipeline
    taxonomy   : dict of label_type → [candidate labels]
                 defaults to the built-in TAXONOMY above

    Returns
    -------
    dict  e.g. {"gender": "Men", "masterCategory": "Footwear",
                "subCategory": "Running Shoes", "confidence": {...}}
    """
    if taxonomy is None:
        taxonomy = TAXONOMY

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    predictions: dict[str, str] = {}
    confidences: dict[str, float] = {}

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for label_type, candidates in taxonomy.items():
            # Build text prompts  →  "a photo of Men's clothing"
            prompts = [f"a photo of {c} fashion product" for c in candidates]
            tokens = clip.tokenize(prompts).to(DEVICE)

            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            sim = (image_features @ text_features.T).squeeze()
            probs = sim.softmax(dim=-1).cpu().numpy()

            best_idx = int(np.argmax(probs))
            predictions[label_type] = candidates[best_idx]
            confidences[label_type] = round(float(probs[best_idx]), 4)

    predictions["confidence"] = confidences  # type: ignore[assignment]
    return predictions


def format_prediction(pred: dict) -> str:
    """Format a classification prediction as a human-readable string."""
    return (
        f"{pred.get('gender', '?')} > "
        f"{pred.get('masterCategory', '?')} > "
        f"{pred.get('subCategory', '?')}"
    )


# ── ResNet50 / EfficientNet fine-tuned head ───────────────────────────────────

class ProductClassifier(nn.Module):
    """
    Lightweight classification head on top of a frozen ImageNet backbone.
    Supports 'resnet50' and 'efficientnet_b0'.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 45,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# preprocessing for ResNet / EfficientNet
RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_with_resnet(
    image_path: str,
    model: ProductClassifier,
    label_map: dict[int, str],
) -> dict[str, str]:
    """
    Run inference with a fine-tuned ResNet / EfficientNet classifier.

    Parameters
    ----------
    image_path : path to product image
    model      : trained ProductClassifier instance
    label_map  : mapping from class index → sub-category name

    Returns
    -------
    {"predicted_class": str, "confidence": float}
    """
    model.eval()
    image = RESNET_TRANSFORM(Image.open(image_path).convert("RGB"))
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = logits.softmax(dim=-1).squeeze().cpu().numpy()

    best_idx = int(np.argmax(probs))
    return {
        "predicted_class": label_map.get(best_idx, "Unknown"),
        "confidence": round(float(probs[best_idx]), 4),
        "all_probs": {label_map.get(i, str(i)): round(float(p), 4)
                      for i, p in enumerate(probs)},
    }


def save_classifier(model: ProductClassifier, path: str = "data/classifier.pt"):
    """Persist a trained ProductClassifier to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Classifier saved to '{path}'")


def load_classifier(
    path: str,
    backbone: str = "resnet50",
    num_classes: int = 45,
) -> ProductClassifier:
    """Load a saved ProductClassifier from disk."""
    model = ProductClassifier(backbone=backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
