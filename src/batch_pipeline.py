from pathlib import Path
import json

import pandas as pd

from src.captioning import load_florence_model, generate_caption
from src.feature_extraction import extract_features_from_caption


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def get_image_paths(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_paths = [
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(image_paths)


def process_images_batch(
    data_dir: str | Path,
    output_json_path: str | Path,
    output_csv_path: str | Path | None = None,
    task_prompt: str = "<MORE_DETAILED_CAPTION>",
) -> list[dict]:
    image_paths = get_image_paths(data_dir)

    if not image_paths:
        raise ValueError(f"No supported image files found in: {data_dir}")

    model, processor, device, torch_dtype = load_florence_model()

    all_results = []

    for image_path in image_paths:
        print(f"Processing: {image_path.name}")

        caption_result = generate_caption(
            image_path=image_path,
            model=model,
            processor=processor,
            device=device,
            torch_dtype=torch_dtype,
            task_prompt=task_prompt,
        )

        features = extract_features_from_caption(caption_result["caption"])

        full_result = {
            "image_path": str(image_path),
            "file_name": image_path.name,
            "task_prompt": caption_result["task_prompt"],
            "user_prompt": caption_result["user_prompt"],
            "final_prompt": caption_result["final_prompt"],
            "caption": caption_result["caption"],
            "features": features,
        }

        all_results.append(full_result)

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved JSON results to: {output_json_path}")

    if output_csv_path is not None:
        rows = []

        for item in all_results:
            row = {
                "file_name": item["file_name"],
                "image_path": item["image_path"],
                "caption": item["caption"],
                "colors": ", ".join(item["features"]["colors"]),
                "garments": ", ".join(item["features"]["garments"]),
                "upper_wear": ", ".join(item["features"]["upper_wear"]),
                "lower_wear": ", ".join(item["features"]["lower_wear"]),
                "footwear": ", ".join(item["features"]["footwear"]),
                "accessories": ", ".join(item["features"]["accessories"]),
                "materials": ", ".join(item["features"]["materials"]),
                "style_words": ", ".join(item["features"]["style_words"]),
                "item_color_pairs": "; ".join(
                    f"{pair['color']} {pair['item']}"
                    for pair in item["features"]["item_color_pairs"]
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)

        print(f"Saved CSV results to: {output_csv_path}")

    return all_results
