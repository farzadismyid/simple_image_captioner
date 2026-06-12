from typing import List, Dict


def evaluate_single_result(result: Dict) -> Dict:
    caption = result.get("caption", "")
    features = result.get("features", {})

    words = caption.split()
    caption_length = len(words)

    garments = features.get("garments", [])
    colors = features.get("colors", [])
    item_color_pairs = features.get("item_color_pairs", [])

    return {
        "file_name": result.get("file_name", ""),
        "caption_length": caption_length,
        "num_garments": len(garments),
        "num_colors": len(colors),
        "num_item_color_pairs": len(item_color_pairs),
        "has_caption": int(len(caption.strip()) > 0),
        "has_pairs": int(len(item_color_pairs) > 0),
    }


