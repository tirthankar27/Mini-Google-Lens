from fastapi import FastAPI, Request, UploadFile, File, Form
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
from backend.retrieval import search_text
from backend.chat import analyze_image, ask_ai
from backend.web_search import search_web_images
from backend.shopping import shopping_links
from backend.category import detect_category

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
    ai_info = None
    shopping = None
    web_images = []

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

        ai_info = analyze_image(
            scene=scene[0]["label"],
            caption=caption,
            ocr_text="\n".join(ocr_text)
        )
        if ai_info["category"] == "unknown":
            ai_info["category"] = detect_category(ai_info["object"])
        primary_prediction = {
            "class": ai_info["object"],
            "confidence": round(scene[0]["confidence"], 2)
        }
        search_query = ai_info.get("search_query", "")

        if not search_query or not search_query.strip():
            search_query = caption

        print("AI INFO:", ai_info)
        print("SEARCH QUERY:", repr(search_query))

        web_images = search_web_images(search_query)

        print("AI INFO:", ai_info)
        print("SEARCH QUERY:", repr(ai_info.get("search_query")))

    else:
        results, info = classify(image)

        similar = search_image(image)

        ai_info = analyze_image(
            scene=None,
            caption=caption,
            ocr_text=""
        )
        ai_info["summary"] = ask_ai(
            question="Describe this object in two concise sentences.",
            scene="",
            caption=caption,
            ocr_text=[],
            prediction=ai_info["object"]
        )
        if ai_info["category"] == "unknown":
            ai_info["category"] = detect_category(ai_info["object"])
        primary_prediction = {
            "class": ai_info["object"],
            "confidence": round(scene[0]["confidence"], 2)
        }
        search_query = ai_info.get("search_query", "")

        if not search_query or not search_query.strip():
            search_query = caption

        print("AI INFO:", ai_info)
        print("SEARCH QUERY:", repr(search_query))

        web_images = search_web_images(search_query)
        print("AI INFO:", ai_info)
        print("SEARCH QUERY:", repr(ai_info.get("search_query")))
        PRODUCT_CATEGORIES = {
            "product",
            "electronics",
            "electronic_device",
            "phone",
            "smartphone",
            "mobile",
            "tablet",
            "laptop",
            "computer",
            "watch",
            "camera",
            "headphones",
            "vehicle",
            "car",
            "motorcycle",
            "shoe",
            "clothing",
            "bag",
            "furniture"
        }
        print(ai_info)
        if ai_info["category"].lower() in PRODUCT_CATEGORIES:

            shopping = shopping_links(ai_info)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "scene": scene,
            "primary_prediction": primary_prediction,
            "top5": results,
            "caption": caption,
            "ocr_text": ocr_text,
            "similar_images": similar,
            "image_path": "/" + file_path,
            "info": info,
            "web_images": web_images,
            "ai_info": ai_info,
            "shopping": shopping,
        }
    )

@app.post("/search")
async def search(
    request: Request,
    query: str = Form(...)
):

    images = search_text(query)

    return templates.TemplateResponse(
        request=request,
        name="search_results.html",
        context={
            "query": query,
            "images": images
        }
    )