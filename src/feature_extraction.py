import re
from collections import OrderedDict


COLOR_WORDS = [
    "black", "white", "blue", "brown", "red", "green", "yellow",
    "pink", "purple", "grey", "gray", "orange", "beige", "cream",
    "navy", "gold", "silver"
]

GARMENT_WORDS = [
    "shirt", "t-shirt", "tee", "top", "blouse", "jacket", "coat",
    "hoodie", "sweater", "jumper", "cardigan", "blazer",
    "jeans", "trousers", "pants", "shorts", "skirt", "dress",
    "shoes", "sneakers", "boots", "heels", "sandals",
    "bag", "handbag", "backpack", "scarf", "hat", "cap", "belt"
]

UPPER_WEAR = [
    "shirt", "t-shirt", "tee", "top", "blouse", "jacket", "coat",
    "hoodie", "sweater", "jumper", "cardigan", "blazer"
]

LOWER_WEAR = [
    "jeans", "trousers", "pants", "shorts", "skirt"
]

FOOTWEAR = [
    "shoes", "sneakers", "boots", "heels", "sandals"
]

ACCESSORIES = [
    "bag", "handbag", "backpack", "scarf", "hat", "cap", "belt"
]

MATERIAL_WORDS = [
    "leather", "denim", "cotton", "wool", "silk", "linen", "suede"
]

STYLE_WORDS = [
    "casual", "formal", "sporty", "elegant", "smart", "classic",
    "minimalist", "streetwear", "vintage", "modern"
]


def _find_terms(text: str, vocabulary: list[str]) -> list[str]:
    text = text.lower()
    found = []

    for word in vocabulary:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, text):
            found.append(word)

    return list(OrderedDict.fromkeys(found))


def extract_features_from_caption(caption: str) -> dict:
    colors = _find_terms(caption, COLOR_WORDS)
    garments = _find_terms(caption, GARMENT_WORDS)
    upper_wear = _find_terms(caption, UPPER_WEAR)
    lower_wear = _find_terms(caption, LOWER_WEAR)
    footwear = _find_terms(caption, FOOTWEAR)
    accessories = _find_terms(caption, ACCESSORIES)
    materials = _find_terms(caption, MATERIAL_WORDS)
    style_words = _find_terms(caption, STYLE_WORDS)

    return {
        "colors": colors,
        "garments": garments,
        "upper_wear": upper_wear,
        "lower_wear": lower_wear,
        "footwear": footwear,
        "accessories": accessories,
        "materials": materials,
        "style_words": style_words,
    }