# Fine-tuning Documentation: Palette Generation Model

## Overview
Fine tuning a  color palette generation model using Google's Gemma-3-270M model with LoRA (Low-Rank Adaptation) in full precision using pytorch native training

Consumes 1GB RAM when serving


## Project Structure
- **train.py**: Main training script with complete fine-tuning pipeline
- **explore_model.ipynb**: Interactive notebook for model exploration and tokenization experiments
- **Data**: CSV file with color palette descriptions and hex values

## Model Architecture

### Base Model
- **Model**: `google/gemma-3-270m` (270M parameters)
- **Type**: Causal Language Model
- **Attention**: Eager implementation for compatibility
- **Precision**: Float32 for stability

### LoRA Configuration
```python
peft_config = LoraConfig(
    r=16,                    # Rank of adaptation
    lora_alpha=32,           # Scaling parameter
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",             # No bias adaptation
    task_type="CAUSAL_LM",   # Causal language modeling
    target_modules=[         # Modules to adapt
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ]
)
```

## Key Insights from explore_model.ipynb

The notebook demonstrates several important concepts:

1. **Tokenizer Exploration**: Shows how the Gemma tokenizer handles special tokens and text encoding
2. **Model Forward Pass**: Demonstrates the difference between inference (no labels) and training (with labels)
3. **Label Masking**: Illustrates how -100 labels prevent loss computation on prompt tokens
4. **Batch Processing**: Shows how custom collation handles variable-length sequences
5. **Generation**: Demonstrates model inference with temperature and top-p sampling


## Data Processing Pipeline

### 1. Data Preprocessing
The training data consists of theme descriptions and corresponding color palettes:

```python
def preprocess(row):
    # Extract colors from CSV
    cols = ast.literal_eval(row["colors"])
    cols = [str(c).lower() for c in cols][:4] + ["#000000"] * max(0, 4 - len(cols))
    
    # Create prompt template
    prompt = (
        "You are a palette generator. Given a theme description, "
        "output ONLY a JSON array of exactly 4 hex colors (lowercase), "
        "no extra text.\n\n"
        f"Description: {row['resp']}\nColors:"
    )
    return prompt, json.dumps(cols)
```



### Text Processing Flow
```
Input: "Fresh greens with a vibrant twist"
                    ↓
Prompt: "You are a palette generator. Given a theme description, 
         output ONLY a JSON array of exactly 4 hex colors (lowercase), 
         no extra text.\n\nDescription: Fresh greens with a vibrant twist.\nColors:"
                    ↓
Colors: ["#c5d70f", "#4dd70f", "#0fd749", "#0fd7c1"]
                    ↓
Full Text: prompt + " " + colors
                    ↓
Tokenization: [14625, 672, 8599, 46128, ...] (token IDs)
                    ↓
Split Point: Identify where colors begin for label masking
```

### Tokenization Example
```
Input Text: "hello there"
                    ↓
Tokenizer Output:
{
    'input_ids': tensor([[14625, 672]]),
    'attention_mask': tensor([[1, 1]])
}
                    ↓
Model Input Shape: [batch_size, sequence_length]
```

## Custom Collation Function

### Purpose
The custom collation function handles:
- **Padding**: Batch sequences to same length
- **Label Masking**: Only compute loss on color tokens
- **Attention Masking**: Ignore padding tokens


### Label Masking Strategy
- **Prompt tokens**: Labels = -100 (ignored in loss)
- **Color tokens**: Labels = actual token IDs (used for loss)
- **Padding tokens**: Labels = -100 (ignored in loss)

### Collation Process Diagram
```
Batch Input:
Sample 1: [prompt_tokens] + [color_tokens] + [pad_tokens]
Sample 2: [prompt_tokens] + [color_tokens] + [pad_tokens]

                    ↓ Custom Collation Function ↓

Output Tensors:
input_ids:     [batch_size, max_length]
attention_mask: [batch_size, max_length]  (1 for real tokens, 0 for padding)
labels:        [batch_size, max_length]   (-100 for prompt/padding, token_ids for colors)

Example:
input_ids:     [[14625, 672, 8599, 46128, 0, 0, 0],
                [14625, 672, 8599, 46128, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 1, 0, 0, 0]]
labels:        [[-100, -100, 8599, 46128, -100, -100, -100],
                [-100, -100, 8599, 46128, -100, -100, -100]]
                ↑      ↑      ↑      ↑      ↑      ↑      ↑
            prompt  prompt  color  color  padding padding padding
            (ignore) (ignore) (train) (train) (ignore) (ignore) (ignore)
```

## Training Configuration

### Hyperparameters
```python
epochs = 3
grad_accum = 16          # Gradient accumulation steps
lr = 5e-5                # Learning rate
max_grad_norm = 1.0      # Gradient clipping
batch_size = 1           # Effective batch size = 1 * 16 = 16
```

### Optimizer & Scheduler
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup (0 warmup steps)
- **Gradient Clipping**: Max norm = 1.0

### Training Loop Features
1. **Gradient Accumulation**: Accumulate gradients over 16 steps
2. **Mixed Precision**: Not used (float32 for stability)
3. **Checkpointing**: Save every 200 steps, keep 2 checkpoints
4. **Evaluation**: Every 100 steps
5. **Logging**: TensorBoard integration

## Training Process Flow

### Training Loop Diagram
```
Start Training
       ↓
   Load Batch
       ↓
   Forward Pass → model(input_ids, attention_mask, labels) → loss
       ↓
   Backward Pass → loss.backward() → gradients
       ↓
   Gradient Accumulation (16 steps)
       ↓
   Gradient Clipping (max_norm=1.0)
       ↓
   Optimizer Step → AdamW update
       ↓
   Scheduler Step → Learning rate update
       ↓
   Zero Gradients
       ↓
   Logging (every 10 steps)
       ↓
   Evaluation (every 100 steps)
       ↓
   Checkpoint (every 200 steps)
       ↓
   Next Batch
```

## Usage Example

### Training
```bash
python train.py
```

### Model Loading
```python
model = AutoModelForCausalLM.from_pretrained("gemma-lora-5/best")
tokenizer = AutoTokenizer.from_pretrained("gemma-lora-5/best")
```

### Inference
```python
prompt = "You are a palette generator...\nDescription: Ocean blues\nColors:"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=64, temperature=0.2)
colors = tokenizer.decode(output[0], skip_special_tokens=True)
```

## Complete Training Pipeline Overview

```
Data Preparation
       ↓
   CSV → Theme Descriptions + Color Palettes
       ↓
   Preprocessing → Prompt Templates + JSON Colors
       ↓
   Tokenization → Token IDs + Split Points
       ↓
   Custom Collation → Batched Tensors + Label Masking
       ↓
   LoRA Model → Efficient Parameter Adaptation
       ↓
   Training Loop → Gradient Accumulation + Optimization
       ↓
   Evaluation → Validation Loss + Sample Generation
       ↓
   Checkpointing → Best Model Selection
       ↓
   Inference → Color Palette Generation
```


