from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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