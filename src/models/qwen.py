from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def get_device() -> str:
    # return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def load_model(model_id: str = MODEL_ID):
    device = get_device()

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "model_id": model_id,
        "model_key": "qwen2_vl",
    }


def generate(
    image_path: str | Path,
    loaded_model: dict,
    task_prompt: str | None = None,
    user_prompt: str | None = None,
    max_new_tokens: int = 128,
    num_beams: int = 1,
) -> dict:

    image = load_image(image_path)

    model = loaded_model["model"]
    processor = loaded_model["processor"]

    if user_prompt is None:
        user_prompt = "Describe the outfit, clothing items, and colors."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return {
        "image_path": str(image_path),
        "model_key": loaded_model["model_key"],
        "model_id": loaded_model["model_id"],
        "task_prompt": task_prompt,
        "user_prompt": user_prompt,
        "final_prompt": user_prompt,
        "caption": generated_text,
        "raw_output": generated_text,
        "parsed_output": {"text": generated_text},
    }
