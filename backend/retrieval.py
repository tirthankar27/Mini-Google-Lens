import os
import pickle
import numpy as np
from PIL import Image

import torch
import open_clip

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading OpenCLIP Retrieval...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

embeddings = np.load("clip_embeddings.npy")

with open("clip_image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

print("CLIP Retrieval Ready")


def _nearest(query_embedding, k=6):

    similarity = embeddings @ query_embedding.T

    indices = np.argsort(-similarity.squeeze())[:k]

    results = []

    for idx in indices:

        path = image_paths[idx]

        relative = os.path.relpath(path, "dataset")

        relative = relative.replace("\\", "/")

        results.append("/dataset/" + relative)

    return results


def search_image(image):

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():

        feature = model.encode_image(image)

        feature /= feature.norm(dim=-1, keepdim=True)

    feature = feature.cpu().numpy()

    return _nearest(feature)


def search_text(text):

    tokens = open_clip.tokenize([text]).to(device)

    with torch.no_grad():

        feature = model.encode_text(tokens)

        feature /= feature.norm(dim=-1, keepdim=True)

    feature = feature.cpu().numpy()

    return _nearest(feature)    