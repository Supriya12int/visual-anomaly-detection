import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist
from datetime import datetime

from cnn_feature_extractor import ResNetFeatureExtractor


# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- LOAD MODEL ONCE --------------------
model = ResNetFeatureExtractor().to(device)
model.eval()


# -------------------- LOAD MEMORY + THRESHOLD --------------------
memory = np.load("normal_patch_memory.npy")
threshold = np.load("threshold.npy")


# -------------------- IMAGE TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# -------------------- MAIN DETECTION FUNCTION --------------------
def detect_anomaly(image_path):
    """
    Runs patch-based anomaly detection on a single image.

    Returns:
        dict with:
        - label
        - score
        - outline_image (path)
        - filled_image (path)
    """

    # -------------------- LOAD IMAGE --------------------
    img = Image.open(image_path).convert("RGB")
    orig = np.array(img.resize((128, 128)))
    img_tensor = transform(img).unsqueeze(0).to(device)

    # -------------------- FEATURE EXTRACTION --------------------
    with torch.no_grad():
        features = model(img_tensor)

    # [1, 512, 4, 4] → [16, 512]
    patches = (
        features.squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, 512)
        .cpu()
        .numpy()
    )

    # -------------------- PATCH DISTANCE --------------------
    distances = cdist(patches, memory, metric="euclidean")
    patch_scores = distances.min(axis=1)

    image_score = float(patch_scores.max())

    # -------------------- DECISION --------------------
    label = "DEFECT" if image_score > threshold else "NORMAL"

    # -------------------- HEATMAP --------------------
    heatmap = patch_scores.reshape(4, 4)
    heatmap_up = cv2.resize(
        heatmap, (128, 128), interpolation=cv2.INTER_NEAREST
    )

    heatmap_up = (heatmap_up - heatmap_up.min()) / (
        heatmap_up.max() - heatmap_up.min() + 1e-8
    )

    mask = (heatmap_up > 0.4).astype(np.uint8) * 255

    # ==================================================
    # OUTPUT 1: OUTLINE IMAGE
    # ==================================================
    outline_img = orig.copy()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(outline_img, contours, -1, (255, 0, 0), 2)

    # ==================================================
    # OUTPUT 2: FILLED IMAGE
    # ==================================================
    filled_img = orig.copy()
    soft_mask = cv2.GaussianBlur(mask, (31, 31), 0)
    soft_mask = soft_mask.astype(np.float32) / 255.0

    highlight = np.zeros_like(orig)
    highlight[:, :, 0] = 255  # red overlay

    for c in range(3):
        filled_img[:, :, c] = (
            orig[:, :, c] * (1 - 0.4 * soft_mask) +
            highlight[:, :, c] * (0.4 * soft_mask)
        )

    # -------------------- SAVE RESULTS (UNIQUE NAMES) --------------------
    os.makedirs("results", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outline_path = f"results/outline_{base_name}_{timestamp}.png"
    filled_path  = f"results/filled_{base_name}_{timestamp}.png"

    cv2.imwrite(outline_path, cv2.cvtColor(outline_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(filled_path, cv2.cvtColor(filled_img, cv2.COLOR_RGB2BGR))

    # -------------------- RETURN --------------------
    return {
        "label": label,
        "score": image_score,
        "outline_image": outline_path,
        "filled_image": filled_path
    }
