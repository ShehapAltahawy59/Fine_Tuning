# evaluation_pipeline.py
import logging
import os
import time
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ===== CONFIG =====
with open('config.yaml') as f:
    config = yaml.safe_load(f)

MODEL_NAME = config['base_model']
OUTPUT_DIR = config["output_dir"]
TUNED_MODEL_PATH = f"{config["output_dir"]}/final_model"  # tuned model folder
DATASET_PATH = "./data_set/final/egypt_pdf_qa_test.jsonl"
MAX_GEN_LENGTH = 2500

# ===== LOAD METRICS =====
exact_match = evaluate.load("exact_match")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# ===== LOAD DATASET =====
dataset = load_dataset("json", data_files={"test": DATASET_PATH})

# ===== LOAD TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def format_propmet(example):
        return {
            "instruction": "Answer the question concisely.",
            "input": example['question'],
            "output": example["answer"]
        }

# ===== EVALUATION FUNCTION =====
def evaluate_model(model, tokenizer, dataset, description="Model"):
    predictions, references = [], []
    total_latency = 0
    
    model.eval()
    
    for row in dataset["test"]:
        answer = row.get("answer", "").strip()
        formatted = format_propmet(row)
        
        # Combine instruction + input into a single text prompt
        prompt = f"{formatted['instruction']}\n{formatted['input']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        
        # Measure latency
        start_time = time.time()
        outputs = model.generate(**inputs, max_length=MAX_GEN_LENGTH)
        latency = time.time() - start_time
        total_latency += latency
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        predictions.append(pred)
        references.append(answer)
    
    # Compute metrics
    em_score = exact_match.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    
    avg_latency = total_latency / len(dataset["test"])
    throughput = len(dataset["test"]) / total_latency if total_latency > 0 else 0
    
    # Print results
    print(f"\n📊 {description} Results:")
    print(f"Exact Match: {em_score}")
    print(f"ROUGE: {rouge_score}")
    print(f"BLEU: {bleu_score}")
    print(f"⚡ Avg Inference Latency: {avg_latency:.3f} sec/request")
    print(f"🚀 Throughput: {throughput:.2f} requests/sec")


def run():
    # ===== LOAD BASE MODEL =====
    print("\nLoading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    evaluate_model(base_model, tokenizer, dataset, description="Base Model")

    # ===== LOAD TUNED MODEL =====
    print("\nLoading Tuned Model...")
    tuned_model = AutoModelForCausalLM.from_pretrained(TUNED_MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    evaluate_model(tuned_model, tokenizer, dataset, description="Tuned Model")
