import torch
import open_clip

device = "cpu"

print("Loading OpenCLIP Router...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.eval()

print("OpenCLIP Router Ready")


candidate_labels = [
    "a printed document",
    "a handwritten letter",
    "a newspaper page",
    "a receipt",
    "a book page",
    "a sheet of music",
    "a frog",
    "a dog",
    "a cat",
    "a bird",
    "a bicycle",
    "a motorcycle",
    "a car",
    "a flower",
    "a tree",
    "a person",
    "a building",
    "a piece of furniture",
    "an electronic device",
    "a plate of food"
]


def detect_scene(image):

    image = preprocess(image).unsqueeze(0)

    text = tokenizer(candidate_labels)

    with torch.no_grad():

        image_features = model.encode_image(image)

        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (
            100.0 *
            image_features @ text_features.T
        )

        probs = similarity.softmax(dim=-1)

    values, indices = probs.topk(5)

    results = []

    for value, idx in zip(values[0], indices[0]):

        results.append({
            "label": candidate_labels[idx],
            "confidence": round(
                value.item()*100,
                2
            )
        })

    return results