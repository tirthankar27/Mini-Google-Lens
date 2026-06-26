import os
import pickle
import numpy as np
from PIL import Image

import torch
import open_clip

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading OpenCLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

dataset_root = "./dataset/256_ObjectCategories"

embeddings = []
image_paths = []

print("Building CLIP embeddings...")

for class_folder in sorted(os.listdir(dataset_root)):

    folder = os.path.join(dataset_root, class_folder)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, file)

        try:

            image = preprocess(
                Image.open(path).convert("RGB")
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                feature = model.encode_image(image)

                feature /= feature.norm(dim=-1, keepdim=True)

            embeddings.append(
                feature.cpu().numpy()[0]
            )

            image_paths.append(path)

        except Exception as e:

            print(path, e)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

print(embeddings.shape)

np.save(
    "clip_embeddings.npy",
    embeddings
)

with open(
    "clip_image_paths.pkl",
    "wb"
) as f:

    pickle.dump(
        image_paths,
        f
    )

print("Done.")