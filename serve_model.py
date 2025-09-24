import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
# add this near your FastAPI app creation
from fastapi.middleware.cors import CORSMiddleware


# ----------------------------
# Config
# ----------------------------
MODEL_DIR = "gemma-lora/best"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model & tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, MODEL_DIR)  # ensure LoRA weights load
model.to(DEVICE)
model.eval()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="Palette Generator API",
    description="Generate JSON color palettes from text descriptions"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*","http://localhost:8000","*"],  # relax as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------
# Request body
# ----------------------------
class DescriptionRequest(BaseModel):
    description: str
    max_new_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.9

# ----------------------------
# Prompt builder (same as training)
# ----------------------------
def build_prompt(description: str) -> str:
    return (
        "You are a palette generator. Given a theme description, output ONLY a JSON array "
        "of exactly 4 hex colors (lowercase), no extra text.\n\n"
        f"Description: {description}\nColors:"
    )

# ----------------------------
# Inference function
# ----------------------------
def generate_palette(description, max_new_tokens=64, temperature=0.2, top_p=0.9):
    prompt = build_prompt(description)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract JSON array
    m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    return m.group(0) if m else "[]"

# ----------------------------
# Routes
# ----------------------------
@app.post("/generate")
def generate(req: DescriptionRequest):
    '''
    sample output: {
                "description": "green → teal → blue → purple gradients",
                    "colors": [
                        "#67c585",
                        "#67c5b5",
                        "#679bc5",
                        "#6f67c5"
                    ]
                    }
    '''
    colors = generate_palette(
        req.description,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p
    )
    return {"description": req.description, "colors": eval(colors)}

@app.get("/")
def home():
    return {"message": "Palette Generator API is running!"}


# uvicorn serve_model:app --host 0.0.0.0 --port 8000 --reload