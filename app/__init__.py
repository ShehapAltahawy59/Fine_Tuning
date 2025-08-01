import os
from fastapi import FastAPI, HTTPException, Header
from huggingface_hub import login
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

import yaml
huggingface_key = os.environ["huggingface_key"]
login(huggingface_key)
with open('config.yaml') as f:
    config = yaml.safe_load(f)
# ===== CONFIG =====
MODEL_PATH = f"{config['output_dir']}/final_model"
MAX_NEW_TOKENS = 100
API_KEY = config['deployment']['api_auth_key']  # Replace with secure key

# ===== LOAD MODEL =====
print("🚀 Loading tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()
print("✅ Model loaded for serving.")
