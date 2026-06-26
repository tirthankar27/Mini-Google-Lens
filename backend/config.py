import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cpu")

# Load classes
with open("classes.json", "r") as f:
    classes = json.load(f)

with open("data/object_info.json", "r") as f:
    object_info = json.load(f)

# Load ResNet
model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.eval()

# Feature extractor
feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

# Embeddings
embeddings = np.load("embeddings.npy")

with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)