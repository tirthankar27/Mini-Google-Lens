from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

print("Loading Qwen...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map=None,
    trust_remote_code=True
)

model.to(device)

print("Qwen Ready")


def ask_ai(
    question,
    scene,
    caption,
    ocr_text,
    prediction=""
):

    context = f"""
You are an AI Vision Assistant.

Scene:
{scene}

Caption:
{caption}

Detected Object:
{prediction}

OCR:
{" ".join(ocr_text)}

Answer ONLY using the information above.
If you are unsure, clearly say so.

User Question:
{question}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful vision assistant."
        },
        {
            "role": "user",
            "content": context
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()

def analyze_image(scene=None, caption="", ocr_text=""):

    prompt = """
You are an expert computer vision assistant.

You receive outputs from multiple AI vision models.

Rules:

1. OCR is ALWAYS the most reliable for documents.

2. BLIP Caption is ALWAYS the most reliable description of an object.

3. OpenCLIP only gives coarse scene understanding and may be wrong.

4. If BLIP identifies a specific object
(for example Apple iPhone, Samsung Galaxy,
Toyota Fortuner, Nike Shoes,
MacBook Air, Coca Cola Bottle etc.)
IGNORE the OpenCLIP prediction completely.

Return ONLY valid JSON.

Format:

{
    "object":"",
    "category":"",
    "search_query":"",
    "summary":""
}
"""

    if scene:
        prompt += f"""
    OpenCLIP Scene:
    {scene}
    """
        prompt += f"""
        BLIP Caption:
        {caption}

        OCR Text:
        {ocr_text}
        """

    messages = [
        {
            "role": "system",
            "content": "You return JSON only."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    try:
        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if match:
            data = json.loads(match.group())

            if not data.get("object"):
                data["object"] = caption

            if not data.get("category"):
                data["category"] = "unknown"

            if not data.get("search_query"):
                data["search_query"] = caption

            if not data.get("summary"):
                data["summary"] = caption

            return data

    except Exception:
        pass

    return {
        "object": caption,
        "category": "unknown",
        "search_query": caption,
        "summary": caption
    }