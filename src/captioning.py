from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "microsoft/Florence-2-base-ft"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_florence_model(model_id: str = MODEL_ID):
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

    return model, processor, device, torch_dtype


def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    return image


def generate_caption(
    image_path: str | Path,
    model,
    processor,
    device: str,
    torch_dtype: torch.dtype,
    prompt: str = TASK_PROMPT,
    max_new_tokens: int = 128,
    num_beams: int = 3,
):
    image = load_image(image_path)

    inputs = processor(
        text=prompt,
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
        task=prompt,
        image_size=(image.width, image.height)
    )

    caption = parsed_answer[prompt]

    return {
        "image_path": str(image_path),
        "prompt": prompt,
        "caption": caption,
        "raw_output": generated_text,
        "parsed_output": parsed_answer,
    }