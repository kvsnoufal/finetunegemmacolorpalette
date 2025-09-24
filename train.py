import os, re, ast, json, math, shutil, random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model



# ----------------------------
# 0) Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------------------
# 1) Load & preprocess data
# ----------------------------
df = pd.read_csv("data/par_dfdata.csv")

def preprocess(row):
    try:
        cols = ast.literal_eval(row["colors"])
    except Exception:
        cols = []
    cols = [str(c).lower() for c in cols][:4] + ["#000000"] * max(0, 4 - len(cols))
    prompt = (
        "You are a palette generator. Given a theme description, output ONLY a JSON array "
        "of exactly 4 hex colors (lowercase), no extra text.\n\n"
        f"Description: {row['resp']}\nColors:"
    )
    return prompt, json.dumps(cols)

data = [preprocess(r) for _, r in df.iterrows()]
train_data, val_data = data[:-2], data[-2:]

# ----------------------------
# 2) Load tokenizer & model
# ----------------------------
model_name = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, attn_implementation="eager", torch_dtype=torch.float32
)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ----------------------------
# 3) Add LoRA
# ----------------------------
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_config)

# ----------------------------
# 4) Torch Dataset & Custom Collator
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, colors = self.samples[idx]
        # Full text = prompt + space + colors
        text = prompt + " " + colors
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        # Find split point (colors start after prompt length)
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"].squeeze(0)
        split_idx = prompt_ids.size(0)  # everything after this is colors
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "split_idx": split_idx
        }

def custom_collate_fn(batch):
    pad_id = tokenizer.pad_token_id
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # ignore by default

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        mask = item["attention_mask"]
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = mask

        # only colors part gets labels
        split_idx = item["split_idx"]
        labels[i, split_idx:L] = ids[split_idx:L]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# ----------------------------
# 5) TensorBoard
# ----------------------------
writer = SummaryWriter(log_dir="./logs5")

# ----------------------------
# 6) Prediction hook
# ----------------------------
def generate_predictions(model, tokenizer, dataset, num_samples=2):
    model.eval()
    outputs = []
    n = min(num_samples, len(dataset))
    for i in range(n):
        prompt, _ = dataset.samples[i]
        inp = prompt + " Colors:"
        inputs = tokenizer(inp, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Only extract colors part
        m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
        colors = m.group(0) if m else "[]"
        outputs.append(f"Prompt: {inp}\nPred: {colors}")
    return outputs

# ----------------------------
# 7) Config & Optimizer
# ----------------------------
epochs = 3
grad_accum = 16
lr = 5e-5
logging_steps = 10
eval_steps = 100
save_steps = 200
save_total_limit = 2
max_grad_norm = 1.0
out_dir = "gemma-lora-5"
os.makedirs(out_dir, exist_ok=True)

steps_per_epoch = math.ceil(len(train_loader)/grad_accum)
total_steps = steps_per_epoch * epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# ----------------------------
# 8) Checkpoint helpers
# ----------------------------
def save_ckpt(path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

saved_ckpts = []
def rotate_ckpts(ckpts, limit):
    while len(ckpts) > limit:
        rm = ckpts.pop(0)
        shutil.rmtree(rm, ignore_errors=True)

# ----------------------------
# 9) Eval function
# ----------------------------
@torch.no_grad()
def evaluate():
    model.eval()
    total = 0.0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
    return total / max(1, len(val_loader))

# ----------------------------
# 10) Training loop
# ----------------------------
global_step = 0
best_val = float("inf")
best_dir = None
accum = 0
loss_accum, loss_count = 0.0, 0
from tqdm import tqdm
for epoch in range(epochs):
    model.train()
    print("epoch: ", epoch)
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / grad_accum
        loss.backward()
        accum += 1
        loss_accum += loss.item()
        loss_count += 1

        if accum % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accum = 0

            if global_step % logging_steps == 0:
                avg_loss = (loss_accum / loss_count) * grad_accum
                writer.add_scalar("train_loss", avg_loss, global_step)
                loss_accum, loss_count = 0.0, 0

            if global_step % eval_steps == 0:
                val_loss = evaluate()
                writer.add_scalar("val_loss", val_loss, global_step)
                preds = generate_predictions(model, tokenizer, val_dataset)
                writer.add_text("sample_preds", "\n\n".join(preds), global_step)
                if val_loss < best_val:
                    best_val = val_loss
                    best_dir = os.path.join(out_dir, "best")
                    if os.path.exists(best_dir):
                        shutil.rmtree(best_dir)
                    os.makedirs(best_dir)
                    save_ckpt(best_dir)

            if global_step % save_steps == 0:
                ckpt = os.path.join(out_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt, exist_ok=True)
                save_ckpt(ckpt)
                saved_ckpts.append(ckpt)
                rotate_ckpts(saved_ckpts, save_total_limit)

    # End-of-epoch eval
    val_loss = evaluate()
    writer.add_scalar("val_loss_epoch", val_loss, global_step)
    if val_loss < best_val:
        best_val = val_loss
        best_dir = os.path.join(out_dir, "best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir)
        save_ckpt(best_dir)

# ----------------------------
# 11) Load best model
# ----------------------------
if best_dir:
    print(f"Loading best model from {best_dir} (val_loss={best_val:.4f})")
    model = AutoModelForCausalLM.from_pretrained(best_dir).to(device)

# Final predictions
final_preds = generate_predictions(model, tokenizer, val_dataset)
writer.add_text("sample_preds_final", "\n\n".join(final_preds), global_step+1)
writer.close()

print("Training complete.")
