# Simple Image Captioning (SIC)

This project is a modular image captioning pipeline built in Python using vision-language models.

The goal is to understand how multimodal models work in practice and to build a clean, extensible system that can later support multiple models, evaluation, and deployment.

---

## What this project does

This project implements an end-to-end pipeline:

1. Load images from a directory  
2. Generate captions using a vision-language model (Florence-2)  
3. Extract structured fashion-related features from captions  
4. Process multiple images in batch  
5. Save outputs in JSON and CSV formats  
6. Support switching between different models (extensible architecture)  

---

## Current capabilities

### Image Captioning
- Uses **Florence-2 (Hugging Face Transformers)**
- Supports prompt-based captioning tasks
- Modular model interface for future models

### Feature Extraction
Extracts structured information from captions:
- colors  
- garments  
- upper wear / lower wear / footwear  
- accessories  
- materials  
- style words  
- **garment-color pairs** (e.g., "blue shirt", "black shoes")  

### Batch Processing
- Processes all images in `data/raw/`  
- Supports:
  - `.png`
  - `.jpg`
  - `.jpeg`
- Generates:
  - `batch_results.json`
  - `batch_results.csv`

### Model Switching (Core Design Feature)
- Central model registry  
- Easy to add new models (e.g., Qwen-VL later)  
- Single interface for all models  

---

## Example output

```json
{
  "image_path": "../data/raw/sample.png",
  "caption": "A man is wearing a blue shirt, a brown jacket, blue jeans, and black shoes.",
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
```

---

## Project structure

```text
SIC/
├── app/
├── data/
│   ├── raw/                # input images
│   └── outputs/            # results (JSON / CSV)
├── notebooks/
│   └── 01_setup_and_first_caption.ipynb
├── src/
│   ├── captioning.py       # model-agnostic interface
│   ├── feature_extraction.py
│   ├── batch_pipeline.py
│   └── models/
│       ├── florence.py     # Florence-2 implementation
│       └── registry.py     # model registry
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

---

## Technologies used

- Python  
- PyTorch  
- Hugging Face Transformers  
- Florence-2 vision-language model  
- Jupyter Notebook (VS Code)  
- uv (package manager)  
- pandas (for CSV export)  

---

## Setup

Clone the repository and install dependencies:

```bash
uv sync
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
notebooks/01_setup_and_first_caption.ipynb
```

---

## Usage

### Single image

```python
from src.captioning import load_caption_model, generate_caption

loaded_model = load_caption_model(model_key="florence2")

result = generate_caption(
    image_path="../data/raw/sample.png",
    loaded_model=loaded_model,
)

print(result["caption"])
```

---

### Batch processing

```python
from src.batch_pipeline import process_images_batch

results = process_images_batch(
    data_dir="../data/raw",
    output_json_path="../data/outputs/batch_results.json",
    output_csv_path="../data/outputs/batch_results.csv",
    model_key="florence2"
)
```

---

## Model switching

Supported models:

```python
from src.models.registry import get_supported_models

print(get_supported_models())
```

Example:

```python
loaded_model = load_caption_model(model_key="florence2")
```

Future models (planned):
- Qwen2-VL  
- Qwen2.5-VL  
- other vision-language models  

---

## Notes

- `pyproject.toml` and `uv.lock` are the main dependency sources  
- `requirements.txt` is generated for compatibility  
- `transformers` is pinned for Florence-2 compatibility  
- Florence-2 uses fixed task prompts (e.g., `<MORE_DETAILED_CAPTION>`)  

---

## Current limitations

- Feature extraction is rule-based  
- Limited understanding of complex descriptions  
- No fine-tuning yet  
- No evaluation metrics implemented yet  
- Florence prompt flexibility is limited for custom instructions  

---

## Next steps

- Add evaluation metrics for captions and features  
- Compare multiple models (Florence vs Qwen)  
- Improve feature extraction (LLM-based parsing)  
- Build a simple user interface (Streamlit or API)  
- Fine-tune models on fashion datasets
- This model will be integrated in fashion compatibility domain 

---

## Project goal

This project is a foundation for building:

- multimodal fashion recommendation systems  
- explainable outfit compatibility models  
- AI systems that combine image and text understanding  

The focus is on clean architecture, modular design, and practical understanding of vision-language models.
