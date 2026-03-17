# Simple Image Captioning (SIC)

This project is a beginner-friendly image captioning pipeline built in Python using Jupyter Notebook and a modular code structure.

The goal is to understand how vision-language models work in practice and how to build a clean, reusable pipeline for future extensions such as fashion understanding, recommendation systems, and multimodal AI applications.

## What this project does

This project implements a simple end-to-end pipeline:

1. Load an input image
2. Generate a caption using a vision-language model (Florence-2)
3. Extract structured fashion-related features from the caption
4. Save results in a reusable format (JSON)

## Current capabilities

- Image captioning using **Florence-2 (Hugging Face Transformers)**
- Modular architecture using `src/`
- Rule-based feature extraction including:
  - colors
  - garments
  - upper/lower wear
  - footwear
  - accessories
  - materials
  - style words
- Extraction of **garment-color pairs** (e.g., "blue shirt", "black shoes")
- Output saving as structured JSON

## Example output

```json
{
  "image_path": "../data/raw/sample.png",
  "caption": "A man is standing on a tiled floor. He is wearing a blue shirt, a brown jacket and blue jeans. He has black shoes on his feet.",
  "features": {
    "colors": ["blue", "brown", "black"],
    "garments": ["shirt", "jacket", "jeans", "shoes"],
    "upper_wear": ["shirt", "jacket"],
    "lower_wear": ["jeans"],
    "footwear": ["shoes"],
    "accessories": [],
    "materials": [],
    "style_words": [],
    "item_color_pairs": [
      {"item": "shirt", "color": "blue"},
      {"item": "jacket", "color": "brown"},
      {"item": "jeans", "color": "blue"},
      {"item": "shoes", "color": "black"}
    ]
  }
}
Project structure
SIC/
├── app/
├── data/
│   ├── raw/
│   └── outputs/
├── notebooks/
│   └── 01_setup_and_first_caption.ipynb
├── src/
│   ├── captioning.py
│   └── feature_extraction.py
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
Technologies used

Python

PyTorch

Hugging Face Transformers

Florence-2 vision-language model

Jupyter Notebook (VS Code)

uv (package manager)

Setup

Clone the repository and install dependencies:

uv sync

Run Jupyter Notebook:

jupyter notebook

Then open:

notebooks/01_setup_and_first_caption.ipynb
Notes

The project uses pyproject.toml and uv.lock as the main dependency source

requirements.txt is generated for compatibility

Transformers is pinned for Florence-2 compatibility

Current limitations

Feature extraction is rule-based and limited to predefined vocabularies

No relation understanding beyond simple color-item pairs

No fine-tuning or domain-specific training yet

Evaluation is not implemented yet

Next steps

Batch processing of multiple images

Evaluation metrics for captions and features

Improved feature extraction (LLM-based or learned models)

Support for alternative models (e.g., Qwen-VL)

Simple deployment using Streamlit or FastAPI

Fine-tuning on fashion datasets