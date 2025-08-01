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
TUNED_MODEL_PATH = f"{config['output_dir']}/final_model"
DATASET_PATH = "./data_set/final/test.jsonl"
MAX_GEN_LENGTH = 512  # Reduced from 2500
MODEL_CACHE = "./model_cache/tinyllama"  # Same cache as training

# ===== LOAD METRICS =====
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# ===== LOAD DATASET =====
dataset = load_dataset("json", data_files={"test": DATASET_PATH})

def load_cached_model():
    """Load model from cache if available, otherwise download"""
    from pathlib import Path
    
    # Check if model exists in cache
    cache_path = Path(MODEL_CACHE)
    config_exists = (cache_path / "config.json").exists()
    model_files = list(cache_path.glob("*.safetensors")) + list(cache_path.glob("*.bin"))
    
    if config_exists and len(model_files) > 0:
        logger.info(f"📁 Loading base model from cache: {MODEL_CACHE}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CACHE,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,
                trust_remote_code=True
            )
            return model
        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Downloading from HuggingFace...")
    
    # Fallback to downloading
    logger.info(f"🌐 Downloading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model

def load_cached_tokenizer():
    """Load tokenizer from cache if available"""
    from pathlib import Path
    
    cache_path = Path(MODEL_CACHE)
    if (cache_path / "tokenizer.json").exists() or (cache_path / "tokenizer_config.json").exists():
        logger.info(f"📁 Loading tokenizer from cache: {MODEL_CACHE}")
        try:
            return AutoTokenizer.from_pretrained(
                MODEL_CACHE,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Tokenizer cache load failed: {e}. Downloading...")
    
    logger.info(f"🌐 Downloading tokenizer: {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ===== LOAD TOKENIZER =====
tokenizer = load_cached_tokenizer()
tokenizer.pad_token = tokenizer.eos_token

def format_prompt(example):
    """Format prompt to match training format"""
    return {
        "prompt": f"Answer this question using the context:\nContext: {example['context']}\nQuestion: {example['question']}",
        "expected": example["answer"]
    }

# ===== EVALUATION FUNCTION =====
def evaluate_model(model, tokenizer, dataset, description="Model"):
    predictions, references = [], []
    total_latency = 0
    
    logger.info(f"Evaluating {description} on {len(dataset)} samples...")
    
    for i, row in enumerate(dataset):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(dataset)}")
            
        formatted = format_prompt(row)
        prompt = formatted['prompt']
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        # Measure latency
        start_time = time.time()
        with torch.no_grad():  # Save memory
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,   # Generate at most 256 new tokens
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,  # Only generate 1 sequence
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        latency = time.time() - start_time
        total_latency += latency
        
        # Extract only the generated part (remove input prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if "Answer:" in generated_text:
            # Split on "Answer:" and take everything after it
            answer_section = generated_text.split("Answer:", 1)[1]
            
            # Find where "Question" begins (if exists)
            question_pos = answer_section.find("Question:")
            if question_pos != -1:
                # Take everything before "Question"
                answer_only = answer_section[:question_pos].strip()
            else:
                # No "Question" found, take the whole answer section
                answer_only = answer_section.strip()
            
            # Clean up by taking only first sentence if period exists
            answer_only = answer_only.split(".")[0].strip()
        else:
            # Fallback if no "Answer:" marker found
            answer_only = generated_text
            
        predictions.append(answer_only)
        references.append(formatted["expected"])
    
    # Clean predictions and references
    predictions = [str(p).strip() for p in predictions if p]
    references = [str(r).strip() for r in references if r]
    
    # Ensure same length
    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]
    
    # Compute metrics
    try:
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        bleu_score = bleu.compute(predictions=predictions, references=references)
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return
    
    # Latency & throughput
    avg_latency = total_latency / len(dataset)
    throughput = len(dataset) / total_latency if total_latency > 0 else 0
    
    # Print results
    logger.info(f"\n📊 {description} Results:")
    logger.info(f"Samples evaluated: {len(predictions)}")
    logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    logger.info(f"BLEU: {bleu_score['bleu']:.4f}")
    logger.info(f"⚡ Avg Inference Latency: {avg_latency:.3f} sec/sample")
    logger.info(f"🚀 Throughput: {throughput:.2f} samples/sec")
    
    return {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'bleu': bleu_score['bleu'],
        'avg_latency': avg_latency,
        'throughput': throughput
    }

def run():
    test_dataset = dataset["test"]
    
    # ===== LOAD BASE MODEL =====
    logger.info("\n🔄 Loading Base Model...")
    base_model = load_cached_model()
    base_results = evaluate_model(base_model, tokenizer, test_dataset, description="Base Model")
    
    # Clean up base model from memory
    del base_model
    torch.cuda.empty_cache()
    
    # ===== LOAD TUNED MODEL =====
    logger.info("\n🔄 Loading Tuned Model...")
    try:
        # Load base model again for LoRA
        base_for_peft = load_cached_model()
        
        # Apply LoRA adapter
        tuned_model = PeftModel.from_pretrained(base_for_peft, TUNED_MODEL_PATH)
        tuned_results = evaluate_model(tuned_model, tokenizer, test_dataset, description="Tuned Model")
        
        # ===== COMPARISON =====
        if base_results and tuned_results:
            logger.info("\n📈 IMPROVEMENT SUMMARY:")
            logger.info(f"Exact Match: {base_results['exact_match']:.4f} → {tuned_results['exact_match']:.4f} ({tuned_results['exact_match']-base_results['exact_match']:+.4f})")
            logger.info(f"ROUGE-L: {base_results['rougeL']:.4f} → {tuned_results['rougeL']:.4f} ({tuned_results['rougeL']-base_results['rougeL']:+.4f})")
            logger.info(f"BLEU: {base_results['bleu']:.4f} → {tuned_results['bleu']:.4f} ({tuned_results['bleu']-base_results['bleu']:+.4f})")
            
    except Exception as e:
        logger.error(f"Error loading tuned model: {e}")
        logger.info("Make sure the tuned model path exists and contains LoRA adapter files")

if __name__ == "__main__":
    run()
