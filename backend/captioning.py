import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cpu"

print("Loading BLIP...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.to(device)
model.eval()

print("BLIP Ready")


def generate_caption(image):

    inputs = processor(
        image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=40
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    ).capitalize()