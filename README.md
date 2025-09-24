# AI-Powered Theme Generation System

A complete end-to-end system that generates color palettes from text descriptions using a fine-tuned Gemma model and applies them to web pages via a Chrome extension.

## 🎯 Project Overview

This project implements a sophisticated AI pipeline that:

1. **Generates synthetic color palette datasets** using color theory algorithms
2. **Labels datasets using GPT-4o** for high-quality theme descriptions
3. **Fine-tunes a Gemma 270M model** using LoRA (Low-Rank Adaptation) for palette generation
4. **Serves the model via FastAPI** for real-time inference
5. **Provides a Chrome extension** that applies generated themes to any webpage

## 🏗️ System Architecture

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
- Gradient clipping and mixed precision support

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

### Dataset Generation Algorithm

The system generates color palettes using color theory principles:

```python
def generate_palette_hex(style="analogous", base_color=(0.5, 0.5, 0.5)):
    """Generate a 4-color palette using color harmony theory"""
    r, g, b = base_color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    if style == "analogous":
        hues = [(h + (i - 1.5) * 0.1) % 1.0 for i in range(4)]
    elif style == "complementary":
        hues = [h, (h + 0.5) % 1.0, (h + 0.1) % 1.0, (h - 0.1) % 1.0]
    # ... more styles
```

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

## 🎨 Color Theory Implementation

The system implements several color harmony algorithms:

### 1. Analogous Colors
Colors adjacent on the color wheel (e.g., blue, blue-green, green, yellow-green)

### 2. Complementary Colors
Colors opposite on the color wheel (e.g., red and cyan)

### 3. Triadic Colors
Three colors evenly spaced on the color wheel (120° apart)

### 4. Monochromatic Colors
Variations of a single hue with different lightness/saturation

### 5. Pastel & Vibrant Styles
Specialized palettes for specific aesthetic preferences

## 🔍 Model Performance

**Training Metrics:**
- **Loss Function**: Cross-entropy on response tokens only
- **Validation**: Separate validation set for model selection
- **Monitoring**: TensorBoard integration for real-time metrics
- **Checkpointing**: Automatic saving of best performing model

**Inference Performance:**
- **Latency**: ~200-500ms per generation (MPS/CUDA)
- **Memory**: ~2GB VRAM for inference
- **Quality**: Coherent 4-color palettes matching descriptions

## 🛠️ Development & Debugging

### TensorBoard Monitoring
```bash
tensorboard --logdir=./logs5
```

### Model Exploration
Use `explore_model.ipynb` for:
- Tokenizer analysis
- Model architecture exploration
- Training data inspection
- Inference testing

### API Testing
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"description": "ocean blues", "temperature": 0.2}'
```

## 📊 Dataset Statistics

- **Total Samples**: 50,000 color palettes
- **Color Styles**: 6 different harmony algorithms
- **Labeling**: GPT-4o generated descriptions
- **Format**: JSON with 4 hex colors per palette
- **Quality**: Human-readable theme descriptions

## 🔒 Security & Privacy

- **API Keys**: Stored in environment variables
- **CORS**: Configured for Chrome extension access
- **Data**: No user data stored or transmitted
- **Local Processing**: All inference happens locally

## 🚀 Future Enhancements

1. **Model Improvements**:
   - Larger base models (Gemma 7B, 70B)
   - Multi-modal input (image + text)
   - Style transfer capabilities

2. **Extension Features**:
   - Theme persistence across sessions
   - Custom color palette import
   - Batch theme application
   - Theme sharing and export

3. **API Enhancements**:
   - Authentication and rate limiting
   - Batch processing endpoints
   - Model versioning
   - A/B testing support
