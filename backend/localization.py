from PIL import Image, ImageDraw

import torch

from transformers import (
    Owlv2Processor,
    Owlv2ForObjectDetection
)

print("Loading OWLv2...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = Owlv2Processor.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
)

model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
).to(device)

print("OWLv2 Ready")


def localize_object(image, label, save_path):

    inputs = processor(
        text=[[label]],
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [image.size[::-1]]
    ).to(device)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.12,
        target_sizes=target_sizes,
        text_labels=[[label]]
    )[0]

    draw = ImageDraw.Draw(image)

    best_score = -1

    best_box = None

    for score, box in zip(
        results["scores"],
        results["boxes"]
    ):

        score = score.item()

        if score > best_score:

            best_score = score

            best_box = box

    if best_box is not None:

        x1, y1, x2, y2 = best_box.tolist()

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5
        )

        draw.text(
            (x1, y1 - 20),
            f"{label} {best_score:.2f}",
            fill="red"
        )

    image.save(save_path)

    return "/" + save_path