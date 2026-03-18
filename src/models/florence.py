from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"

FIXED_TASK_PROMPTS = {
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OCR>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_SEGMENTATION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
    "<OCR_WITH_REGION>",
}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def build_prompt(
    task_prompt: str = DEFAULT_TASK_PROMPT,
    user_prompt: str | None = None,
) -> str:
    if task_prompt in FIXED_TASK_PROMPTS:
        return task_prompt

    if user_prompt and user_prompt.strip():
        return f"{task_prompt} {user_prompt.strip()}"

    return task_prompt


def load_model(model_id: str = MODEL_ID):
    device = get_device()
    torch_dtype = get_torch_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "torch_dtype": torch_dtype,
        "model_id": model_id,
        "model_key": "florence2",
    }


def generate(
    image_path: str | Path,
    loaded_model: dict,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    user_prompt: str | None = None,
    max_new_tokens: int = 128,
    num_beams: int = 3,
) -> dict:
    image = load_image(image_path)
    final_prompt = build_prompt(task_prompt=task_prompt, user_prompt=user_prompt)

    model = loaded_model["model"]
    processor = loaded_model["processor"]
    device = loaded_model["device"]
    torch_dtype = loaded_model["torch_dtype"]

    inputs = processor(
        text=final_prompt,
        images=image,
        return_tensors="pt"
    ).to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    caption = parsed_answer.get(task_prompt, "")

    return {
        "image_path": str(image_path),
        "model_key": loaded_model["model_key"],
        "model_id": loaded_model["model_id"],
        "task_prompt": task_prompt,
        "user_prompt": user_prompt,
        "final_prompt": final_prompt,
        "caption": caption,
        "raw_output": generated_text,
        "parsed_output": parsed_answer,
    }
