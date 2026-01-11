import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist

from cnn_feature_extractor import ResNetFeatureExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetFeatureExtractor().to(device)
model.eval()

memory = np.load("normal_patch_memory.npy")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

NORMAL_DIR = "../dataset/test/normal"

scores = []

for img_name in os.listdir(NORMAL_DIR):
    img_path = os.path.join(NORMAL_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    patches = (
        features.squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, 512)
        .cpu()
        .numpy()
    )

    distances = cdist(patches, memory, metric="euclidean")
    patch_scores = distances.min(axis=1)

    scores.append(patch_scores.max())

scores = np.array(scores)

threshold = scores.mean() + 3 * scores.std()
np.save("threshold.npy", threshold)

print("Threshold saved:", threshold)
