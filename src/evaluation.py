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


def evaluate_batch_results(results: List[Dict]) -> Dict:
    if not results:
        return {"error": "No results provided"}

    evaluated = [evaluate_single_result(r) for r in results]

    total = len(evaluated)

    avg_caption_length = sum(r["caption_length"] for r in evaluated) / total
    avg_garments = sum(r["num_garments"] for r in evaluated) / total
    avg_colors = sum(r["num_colors"] for r in evaluated) / total
    avg_pairs = sum(r["num_item_color_pairs"] for r in evaluated) / total

    captions_present = sum(r["has_caption"] for r in evaluated)
    pairs_present = sum(r["has_pairs"] for r in evaluated)

    return {
        "total_images": total,
        "avg_caption_length": round(avg_caption_length, 2),
        "avg_garments": round(avg_garments, 2),
        "avg_colors": round(avg_colors, 2),
        "avg_item_color_pairs": round(avg_pairs, 2),
        "caption_success_rate": round(captions_present / total, 2),
        "pair_detection_rate": round(pairs_present / total, 2),
    }
