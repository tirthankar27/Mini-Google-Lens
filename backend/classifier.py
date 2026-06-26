import torch
from torchvision import transforms
from PIL import Image

from backend.config import model, classes
from backend.config import model, classes, object_info

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def clean_name(name):
    return name.split(".",1)[1].replace("-", " ").title()


def classify(image: Image.Image):

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(outputs,1)

        top_probs, top_classes = torch.topk(
            probs,
            5
        )

    results = []

    for prob, cls in zip(
        top_probs[0],
        top_classes[0]
    ):

        results.append({
            "class": clean_name(
                classes[cls.item()]
            ),
            "confidence": round(
                prob.item()*100,
                2
            )
        })

    top_prediction = results[0]["class"]

    info = object_info.get(
        top_prediction,
        {
            "category": "Unknown",
            "description": "Information not available."
        }
    )

    return results, info