# Multimodal Document Extraction Agent

A LangChain vision pipeline that extracts structured data from images
and PDFs using llama.cpp's multimodal endpoint — no OCR, no separate
vision API, just a single local model.

## Problem

Extracting structured data from scanned invoices, receipts, forms, or
PDFs typically requires: (a) a separate OCR engine, (b) a cloud vision
API, or (c) fragile template-based parsing. Layout, tables, and
formatting are easily lost.

## Solution

llama.cpp's `--mmproj` multimodal projection loads a vision encoder
alongside a GGUF language model, giving the model direct access to
image pixel data via llama.cpp's `/v1/chat/completions` endpoint.
This agent wraps that pipeline: load image/PDF → encode as base64
data URI → send to vision model → parse validated Pydantic output.

## Quick Start

```python
from pydantic import BaseModel, Field
from extraction_chain import MultimodalExtractionChain
from loaders import ImageLoader

class Invoice(BaseModel):
    vendor: str = Field(description="Vendor name")
    total: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")

# Load images from a folder
loader = ImageLoader()
images = loader.load_folder("invoices/")

# Create extraction chain
chain = MultimodalExtractionChain(
    base_url="http://127.0.0.1:8080",
    schema=Invoice,
    system_prompt="Extract invoice fields from the image.",
)

# Extract from each image
for img in images:
    result = chain.extract(img.data_uri)
    print(result)
```

## Features

- **Image loading** — PNG, JPEG, GIF, WebP, BMP; auto-resize to fit
  `--image-max-tokens` budget; base64 data URIs (no file serving needed)
- **PDF loading** — renders pages to images via `pdf2image` (poppler);
  fallback to placeholder if pdf2image not installed
- **Vision ChatModel** — LangChain `BaseChatModel` subclass sending
  OpenAI-compatible multimodal messages to `/v1/chat/completions`
- **Structured extraction** — `MultimodalExtractionChain` returns
  validated Pydantic models (or raw text)
- **Batch & context** — `extract_batch()` for multiple images;
  `extract_with_context()` for page-level context
- **Media server config** — optional `MediaServerConfig` for file:// URL
  serving via `--media-path`

## Starting the Server

You need a **GGUF model with multimodal support** and the corresponding
`mmproj` file. Use the `-hf` flag to auto-download both:

```bash
llama-server \
    -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
    --host 127.0.0.1 --port 8080
```

The `mmproj` file is auto-detected and downloaded alongside the model.

> **Note:** Only models with `"vision"` in their capabilities support
> image inputs. Verify with:
> ```bash
> curl -s http://localhost:8080/v1/model/info | python3 -m json.tool
> ```

### Launcher Dashboard

The `tools/llama-server-launcher.html` dashboard provides a visual
interface with a pre-configured **Vision** mode — pick from 9
multimodal models (Gemma 4, Qwen2.5 VL, SmolVLM2, Pixtral, Qwen2.5
Omni) and auto-generate the correct start command.

```bash
open tools/llama-server-launcher.html
```

## Project Structure

```
2.3-multimodal-doc-agent/
├── loaders/
│   ├── __init__.py
│   ├── image_loader.py       # Image → base64 data URI
│   └── pdf_loader.py         # PDF page → PNG → data URI
├── extraction_chain/
│   ├── __init__.py
│   ├── vision_model.py       # VisionChatModel (BaseChatModel)
│   └── chain.py              # MultimodalExtractionChain
├── media_server_config/
│   ├── __init__.py
│   └── config.py             # --media-path CLI flags
├── eval/                     # Evaluation results
├── tests/
│   ├── test_loaders.py
│   └── test_extraction_chain.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
