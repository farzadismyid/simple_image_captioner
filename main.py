import argparse
from pathlib import Path

from src.batch_pipeline import process_images_batch
from src.models.registry import get_supported_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch image captioning and feature extraction."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to the input image directory."
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="data/outputs/batch_results.json",
        help="Path to save batch JSON results."
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/outputs/batch_results.csv",
        help="Path to save batch CSV results."
    )

    parser.add_argument(
        "--summary-json",
        type=str,
        default="data/outputs/batch_summary.json",
        help="Path to save batch summary JSON."
    )

    parser.add_argument(
        "--model-key",
        type=str,
        default="florence2",
        choices=get_supported_models(),
        help="Model key to use for caption generation."
    )

    parser.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="Optional task prompt for supported models."
    )

    return parser.parse_args()



