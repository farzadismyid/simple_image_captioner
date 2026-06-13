from src.models.registry import get_model_entry


def load_caption_model(model_key: str = "florence2") -> dict:
    model_entry = get_model_entry(model_key)
    return model_entry["load_model"]()


def generate_caption(
    image_path: str,
    loaded_model: dict,
    model_key: str = "florence2",
    task_prompt: str | None = None,
    user_prompt: str | None = None,
    max_new_tokens: int = 128,
    num_beams: int = 3,
) -> dict:
    model_entry = get_model_entry(model_key)

    if task_prompt is None:
        task_prompt = model_entry["default_task_prompt"]

    return model_entry["generate"](
        image_path=image_path,
        loaded_model=loaded_model,
        task_prompt=task_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
