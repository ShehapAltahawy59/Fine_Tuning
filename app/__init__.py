from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

with open('config.yaml') as f:
    config = yaml.safe_load(f)
# ===== CONFIG =====
MODEL_PATH = "./models/mistral_egypt_latest"
MAX_NEW_TOKENS = 2000
API_KEY = config['api_auth_key']  # Replace with secure key

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
