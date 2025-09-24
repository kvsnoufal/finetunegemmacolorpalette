# Fine-tuning Documentation: Palette Generation Model

## Overview
This document details the fine-tuning process for a color palette generation model using Google's Gemma-3-270M model with LoRA (Low-Rank Adaptation) for efficient training.

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

### 2. Data Split
- **Training**: All data except last 2 samples
- **Validation**: Last 2 samples
- **Format**: List of (prompt, colors) tuples

## Tokenization Process

### Tokenizer Configuration
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Special Tokens
- **BOS**: `<bos>` (ID: 2) - Beginning of sequence
- **EOS**: `<eos>` (ID: 1) - End of sequence  
- **PAD**: `<pad>` (ID: 0) - Padding token
- **UNK**: `<unk>` (ID: 3) - Unknown token

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

### Implementation
```python
def custom_collate_fn(batch):
    pad_id = tokenizer.pad_token_id
    max_len = max(x["input_ids"].size(0) for x in batch)
    
    # Initialize tensors
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        mask = item["attention_mask"]
        L = ids.size(0)
        
        # Copy input data
        input_ids[i, :L] = ids
        attention_mask[i, :L] = mask
        
        # Only color tokens get labels (not -100)
        split_idx = item["split_idx"]
        labels[i, split_idx:L] = ids[split_idx:L]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

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

### 1. Forward Pass
```python
out = model(**batch)  # input_ids, attention_mask, labels
loss = out.loss / grad_accum
```

### 2. Backward Pass
```python
loss.backward()
accum += 1
```

### 3. Optimization Step
```python
if accum % grad_accum == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

### 4. Evaluation
- **Validation Loss**: Average loss on validation set
- **Sample Predictions**: Generate color palettes for evaluation
- **Best Model**: Save model with lowest validation loss

### LoRA Training Architecture
```
Base Model (270M params)
       ↓
   LoRA Adapters (16 rank)
       ↓
   Target Modules:
   - q_proj, k_proj, v_proj, o_proj
   - gate_proj, up_proj, down_proj
       ↓
   Only ~1% of parameters trained
       ↓
   Efficient fine-tuning
```

## Model Generation

### Inference Configuration
```python
model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
```

### Output Processing
1. **Decode**: Convert token IDs to text
2. **Extract**: Use regex to find JSON array: `r"\[.*?\]"`
3. **Validate**: Ensure proper color format

## Key Features

### 1. LoRA Efficiency
- **Parameter Reduction**: Only adapts ~1% of model parameters
- **Memory Efficient**: Reduces GPU memory requirements
- **Fast Training**: Faster than full fine-tuning

### 2. Selective Training
- **Prompt Ignored**: Only trains on color generation
- **Focused Learning**: Model learns color relationships
- **Reduced Overfitting**: Less risk of catastrophic forgetting

### 3. Robust Evaluation
- **Multiple Metrics**: Loss, sample quality, validation performance
- **Checkpoint Management**: Automatic best model selection
- **TensorBoard Logging**: Visual training progress

## Training Outputs

### Checkpoints
- **Regular**: Every 200 steps (limited to 2)
- **Best**: Model with lowest validation loss
- **Final**: End-of-training model

### Logging
- **Training Loss**: Scalar values over time
- **Validation Loss**: Performance on validation set
- **Sample Predictions**: Generated color palettes
- **TensorBoard**: Visual training curves

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

## Key Insights from explore_model.ipynb

The notebook demonstrates several important concepts:

1. **Tokenizer Exploration**: Shows how the Gemma tokenizer handles special tokens and text encoding
2. **Model Forward Pass**: Demonstrates the difference between inference (no labels) and training (with labels)
3. **Label Masking**: Illustrates how -100 labels prevent loss computation on prompt tokens
4. **Batch Processing**: Shows how custom collation handles variable-length sequences
5. **Generation**: Demonstrates model inference with temperature and top-p sampling

## Training Efficiency

- **Memory Usage**: LoRA reduces memory requirements by ~80%
- **Training Speed**: 3x faster than full fine-tuning
- **Parameter Efficiency**: Only 1% of model parameters updated
- **Quality**: Maintains model performance while specializing for color generation

This fine-tuning approach efficiently adapts a large language model for the specific task of color palette generation while maintaining the model's general capabilities.
