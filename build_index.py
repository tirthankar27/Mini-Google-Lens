import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load classes
with open("classes.json", "r") as f:
    classes = json.load(f)

# Load trained model
model = models.resnet50()
model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

model = model.to(device)
model.eval()

# Remove classifier layer
feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

dataset_root = "./dataset/256_ObjectCategories"

embeddings = []
image_paths = []

print("Building embeddings...")

for class_folder in os.listdir(dataset_root):

    folder_path = os.path.join(
        dataset_root,
        class_folder
    )

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):

        if not img_name.endswith(".jpg"):
            continue

        path = os.path.join(
            folder_path,
            img_name
        )

        try:

            image = Image.open(path).convert("RGB")

            tensor = (
                transform(image)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():

                feature = (
                    feature_extractor(tensor)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            embeddings.append(feature)
            image_paths.append(path)

        except:
            continue

embeddings = np.array(
    embeddings,
    dtype=np.float32
)
np.save("embeddings.npy", embeddings)

print("Embedding shape:", embeddings.shape)

with open(
    "image_paths.pkl",
    "wb"
) as f:

    pickle.dump(
        image_paths,
        f
    )

print("Embeddings and image paths saved!")