from src.models.registry import get_model_entry


def load_caption_model(model_key: str = "florence2") -> dict:
    model_entry = get_model_entry(model_key)
    return model_entry["load_model"]()


