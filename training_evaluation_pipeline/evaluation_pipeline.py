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
    
    for row in dataset:
        formatted = format_propmet(row)
        prompt = f"{formatted['instruction']}\n{formatted['input']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        
        # Measure latency
        start_time = time.time()
        outputs = model.generate(
                    **inputs,
                    max_length=5000,
                    temperature=0.7,       # Controls randomness (0.0-1.0+, lower = more deterministic)
                    top_k=50,             # Consider top k most probable tokens at each step
                    top_p=0.95,           # Nucleus sampling - consider smallest set with cumulative prob >= p
                    do_sample=True,       # Enable sampling (required for temp/top_k/top_p)
                    num_return_sequences=2 # Number of sequences to return
                )
        latency = time.time() - start_time
        total_latency += latency
        prompt_length = len(inputs.input_ids[0])
        pred_full = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generated_text = pred_full[len(prompt):].strip()
        
        predictions.append(generated_text)
        references.append(formatted["output"])
    
    # Clean predictions and references
    predictions = [str(p).strip() for p in predictions]
    references = [str(r).strip() for r in references]
    
    # Metrics
    em_score = exact_match.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    
    # Latency & throughput
    avg_latency = total_latency / len(dataset)
    throughput = len(dataset) / total_latency if total_latency > 0 else 0
    
    # Print results
    logger.info(f"\n📊 {description} Results:")
    logger.info(f"Exact Match: {em_score}")
    logger.info(f"ROUGE: {rouge_scores}")
    logger.info(f"BLEU: {bleu_score}")
    logger.info(f"⚡ Avg Inference Latency: {avg_latency:.3f} sec/request")
    logger.info(f"🚀 Throughput: {throughput:.2f} requests/sec")


def run():
    # ===== LOAD BASE MODEL =====
    logger.info("\nLoading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    evaluate_model(base_model, tokenizer, dataset, description="Base Model")

    # ===== LOAD TUNED MODEL =====
    logger.info("\nLoading Tuned Model...")
    tuned_model = AutoModelForCausalLM.from_pretrained(TUNED_MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    evaluate_model(tuned_model, tokenizer, dataset, description="Tuned Model")
