from src.models import florence


MODEL_REGISTRY = {
    "florence2": {
        "load_model": florence.load_model,
        "generate": florence.generate,
        "default_task_prompt": florence.DEFAULT_TASK_PROMPT,
    },
     future:
     "qwen2_vl": {
         "load_model": qwen.load_model,
         "generate": qwen.generate,
         "default_task_prompt": "...",
     },
}


def get_supported_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_entry(model_key: str) -> dict:
    if model_key not in MODEL_REGISTRY:
        supported = ", ".join(get_supported_models())
        raise ValueError(
            f"Unsupported model_key '{model_key}'. Supported models: {supported}"
        )
    return MODEL_REGISTRY[model_key]
