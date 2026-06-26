import numpy as np
import torch

from torchvision import transforms

from backend.config import (
    feature_extractor,
    embeddings,
    image_paths
)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def find_similar(image):

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        feature = (
            feature_extractor(tensor)
            .squeeze()
            .numpy()
            .astype(np.float32)
        )

    distances = np.linalg.norm(
        embeddings-feature,
        axis=1
    )

    nearest = np.argsort(distances)[:6]

    images = []

    for idx in nearest[1:6]:

        path = image_paths[idx]

        path = "/" + path.lstrip("./")

        images.append(path)

    return images