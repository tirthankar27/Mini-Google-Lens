from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from PIL import Image
import shutil

from backend.classifier import classify
from backend.similarity import find_similar
from backend.captioning import generate_caption
from backend.ocr import extract_text
from backend.clip_router import detect_scene
from backend.retrieval import search_image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    file_path = f"static/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path).convert("RGB")

    scene = detect_scene(image)

    scene_label = scene[0]["label"]

    caption = generate_caption(image)

    results = None
    info = None
    ocr_text = []
    similar = []

    DOCUMENT_KEYWORDS = [
        "document",
        "letter",
        "receipt",
        "newspaper",
        "book",
        "sheet music"
    ]

    is_document = any(
        word in scene_label.lower()
        for word in DOCUMENT_KEYWORDS
    )

    if is_document:
        ocr_text = extract_text(file_path)

    else:
        results, info = classify(image)
        similar = search_image(image)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "scene": scene,
            "prediction": results[0] if results else None,
            "top5": results,
            "caption": caption,
            "ocr_text": ocr_text,
            "similar_images": similar,
            "image_path": "/" + file_path,
            "info": info
        }
    )