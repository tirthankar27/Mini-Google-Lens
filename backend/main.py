from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json


app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
app.mount(
    "/dataset",
    StaticFiles(directory="dataset"),
    name="dataset"
)

templates = Jinja2Templates(directory="templates")
# Load classes
with open("classes.json", "r") as f:
    classes = json.load(f)

def clean_class_name(name):
    return name.split(".", 1)[1].replace("-", " ").title()

# Device
device = torch.device("cpu")

# Load trained model
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

embeddings = np.load("embeddings.npy")

with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

model.eval()

feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    file_path = f"static/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )
    image = Image.open(file_path).convert("RGB")

    img_tensor = (
        transform(image)
        .unsqueeze(0)
    )

    with torch.no_grad():

        outputs = model(img_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        top_probs, top_classes = torch.topk(
            probs,
            5
        )
        query_embedding = (
            feature_extractor(img_tensor)
            .squeeze()
            .numpy()
            .astype(np.float32)
        )
    distances = np.linalg.norm(
        embeddings - query_embedding,
        axis=1
    )

    nearest = np.argsort(distances)[:6]

    similar_images = []

    for idx in nearest[1:6]:

        path = image_paths[idx]

        relative_path = "/" + path.lstrip("./")

        similar_images.append(
            relative_path
        )

    results = []

    for prob, cls in zip(top_probs[0], top_classes[0]):
        results.append({
            "class":
            clean_class_name(
                classes[cls.item()]
            ),
            "confidence":
            round(
                prob.item() * 100,
                2
            )
        })

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "prediction": results[0],
            "top5": results,
            "image_path": "/" + file_path,
            "similar_images": similar_images
        }
    )