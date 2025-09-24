# Fientuning Gemma 3 270M for Color Theme Generation

A complete end-to-end system that generates color palettes from text descriptions using a fine-tuned Gemma model and applies them to web pages via a Chrome extension.

See [`finetuningdoc.md`](finetuningdoc.md) for detailed model training details.


> **Demo Video:**  

> [![Watch the demo](/docs/gemma1.gif)](/docs/gemma1.gif)

## 🎯 Project Overview

This project implements a pipeline that:

1. **Generates synthetic color palette datasets** 
2. **Labels datasets using GPT-4o** for high-quality theme descriptions
3. **Fine-tunes a Gemma 270M model** using LoRA (Low-Rank Adaptation) for palette generation in full precision
4. **Serves the model via FastAPI** for real-time inference
5. **Provides a Chrome extension** that applies generated themes to any webpage

## 🏗️ Steps

```
Dataset Generation → GPT-4o Labeling → Model Training → FastAPI Serving → Chrome Extension
     ↓                    ↓                ↓              ↓              ↓
Color Palettes    Theme Descriptions   Fine-tuned    REST API      Web Page
(50k samples)     (Parallel Processing)   Model      Endpoint      Theming
```

## 📁 Project Structure

```
clean_dir/
├── generate_dataset.py          # Color palette dataset generation
├── label_dataset_with_openai.py # GPT-4o labeling with parallel processing
├── train.py                     # Gemma model fine-tuning with LoRA
├── serve_model.py               # FastAPI model serving
├── explore_model.ipynb          # Model exploration and experimentation
├── theme-chrome/                # Chrome extension
│   ├── manifest.json           # Extension configuration
│   ├── popup.html              # Extension UI
│   ├── popup.js                # Extension logic
│   ├── popup.css               # Extension styling
│   ├── contentScript.js        # Web page CSS injection
│   ├── background.js           # Extension service worker
│   └── icons/                  # Extension icons
└── data/                       # Generated datasets
    ├── palettes.json           # Raw color palettes
    └── par_dfdata.csv          # Labeled dataset
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers peft pandas numpy fastapi uvicorn litellm python-dotenv
```

### 1. Dataset Generation

Generate 50,000 color palettes using color theory algorithms:

```bash
python generate_dataset.py
```

**Features:**
- Multiple color harmony styles (analogous, complementary, triadic, monochromatic)
- HSL color space manipulation for consistent palettes
- JSON output format for easy processing

### 2. Dataset Labeling

Label palettes with GPT-4o descriptions:

```bash
python label_dataset_with_openai.py
```

**Features:**
- Parallel processing with configurable workers (default: 5)
- Automatic saving every 20 processed samples
- Structured JSON output with Pydantic validation
- Error handling and retry logic

### 3. Model Training

See [`finetuningdoc.md`](finetuningdoc.md) for detailed model training details.

Fine-tune Gemma 270M with LoRA:

```bash
python train.py
```

**Training Configuration:**
- **Model**: Google Gemma 3 270M
- **Method**: LoRA (Low-Rank Adaptation) with r=16, alpha=32
- **Precision**: Full float32 training
- **Optimizer**: AdamW with linear warmup
- **Batch Size**: 1 with gradient accumulation (16)
- **Learning Rate**: 5e-5
- **Epochs**: 3

**Key Features:**
- Custom data collator for prompt/response separation
- TensorBoard logging for monitoring
- Automatic checkpointing and model selection
- Gradient clipping

### 4. Model Serving

Start the FastAPI server:

```bash
uvicorn serve_model:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**
- `POST /generate` - Generate color palette from description
- `GET /` - Health check

**Example Request:**
```json
{
  "description": "muted desert sunset",
  "max_new_tokens": 64,
  "temperature": 0.2,
  "top_p": 0.9
}
```

**Example Response:**
```json
{
  "description": "muted desert sunset",
  "colors": ["#d4a574", "#c49b6a", "#b89160", "#ac8756"]
}
```

### 5. Chrome Extension

Load the extension in Chrome:

1. Open Chrome → Extensions → Developer mode
2. Click "Load unpacked" → Select `theme-chrome/` folder
3. Pin the extension to toolbar

**Usage:**
1. Click the extension icon
2. Enter a theme description (e.g., "ocean blues", "autumn leaves")
3. Click "Generate & Apply"
4. Watch the webpage transform with the new color scheme

## 🔧 Technical Details



### Model Architecture

**Base Model**: Google Gemma 3 270M
- **Architecture**: Transformer-based causal language model
- **Parameters**: 270M
- **Context Length**: 8192 tokens
- **Fine-tuning**: LoRA adaptation on attention layers

**LoRA Configuration:**
```python
peft_config = LoraConfig(
    r=16,                    # Rank of adaptation
    lora_alpha=32,           # Scaling parameter
    lora_dropout=0.05,       # Dropout rate
    bias="none",             # No bias adaptation
    task_type="CAUSAL_LM",   # Causal language modeling
    target_modules=[         # Target attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Training Process

**Data Format:**
```
Prompt: "You are a palette generator. Given a theme description, output ONLY a JSON array of exactly 4 hex colors (lowercase), no extra text.\n\nDescription: {theme}\nColors:"
Response: ["#color1", "#color2", "#color3", "#color4"]
```

**Training Features:**
- **Pytorch native training** - no hf trainer apis
- **Custom Collator**: Separates prompt from response for targeted learning
- **Label Masking**: Only response tokens contribute to loss
- **Gradient Accumulation**: Simulates larger batch sizes
- **Learning Rate Scheduling**: Linear warmup and decay
- **Checkpointing**: Saves best model based on validation loss

### Chrome Extension Architecture

**Manifest V3** with the following components:

1. **Popup Interface** (`popup.html/js`):
   - Theme description input
   - Generate & Apply button
   - Color swatch display
   - Status feedback

2. **Background Service Worker** (`background.js`):
   - API communication with FastAPI server
   - Message routing between popup and content scripts
   - Tab management

3. **Content Script** (`contentScript.js`):
   - CSS injection into web pages
   - Color palette application
   - DOM manipulation for theme application

**CSS Injection Strategy:**
```css
:root.palette-painter {
  --pp-c0: #primary-color;
  --pp-c1: #secondary-color;
  --pp-c2: #accent-color;
  --pp-c3: #highlight-color;
  --pp-on-c0: #text-on-primary;
  /* ... more variables */
}

/* Global overrides for all elements */
:root.palette-painter body,
:root.palette-painter body *:not(img):not(video) {
  background-color: var(--pp-c0) !important;
  color: var(--pp-on-c0) !important;
  /* ... more overrides */
}
```

## 🛠️ Development & Debugging


### Model Exploration
Use `explore_model.ipynb` for:
- Tokenizer analysis
- Model architecture exploration
- Training data inspection
- Inference testing
