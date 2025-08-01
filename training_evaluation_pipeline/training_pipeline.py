# training_pipeline.py
import logging
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import wandb
import yaml
from dotenv import load_dotenv
from huggingface_hub import login

# Initialize logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
# Authenticate
huggingface_key = os.environ["huggingface_key"]
login(huggingface_key)
wandb_key = os.environ["wandb_key"]
wandb.login(key=wandb_key)

# Configurations
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = config["output_dir"]
RESUME_CHECKPOINT = None  # Set path to resume or None
MAX_LENGTH = 1024  # TinyLlama optimal context
MODEL_CACHE = "./model_cache/tinyllama"
Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)

def format_chat_prompt(row):
    """Format data for TinyLlama chat template"""
    return [
        {
            "role": "user", 
            "content": f"Answer this question using the context:\nContext: {row['context']}\nQuestion: {row['question']}"
        },
        {
            "role": "assistant", 
            "content": row["answer"]
        }
    ]

def tokenize_function(examples, tokenizer):
    """Tokenize with proper chat formatting"""
    texts = []
    for messages in examples["messages"]:
        # Handle cases where chat template might not exist
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except:
            # Fallback formatting
            text = f"User: {messages[0]['content']}\nAssistant: {messages[1]['content']}"
        texts.append(text)
    
    return tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding=False,  # Let data collator handle padding
        truncation=True,
        return_tensors=None  # Return lists, not tensors
    )


def load_or_download_model():
    """Load model from cache or download and cache it"""
    Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Check if model already exists in cache
    required_files = ["config.json", "model.safetensors"]
    cache_complete = all((Path(MODEL_CACHE)/file).exists() for file in required_files)

    try:
        if cache_complete:
            print("Loading model from local cache...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CACHE,
                device_map="auto",
                quantization_config=bnb_config,
                local_files_only=True,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CACHE,
                local_files_only=True,
                trust_remote_code=True
            )
        else:
            print("Downloading and caching model...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
            # Save to cache
            model.save_pretrained(MODEL_CACHE, safe_serialization=True)
            tokenizer.save_pretrained(MODEL_CACHE)
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to non-cached version if cache is corrupted
        print("Attempting direct load from Hugging Face...")
        return (
            AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            ),
            AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
        )

def run():
    # ===== LOAD DATASET =====
    dataset = load_dataset("json", data_files={
        "train": "./data_set/final/egypt_pdf_qa_train.jsonl",
        "validation": "./data_set/final/egypt_pdf_qa_val.jsonl",
        "test": "./data_set/final/egypt_pdf_qa_test.jsonl"
    })

    # Apply chat formatting
    dataset = dataset.map(lambda x: {"messages": format_chat_prompt(x)})

    # ===== LOAD MODEL AND TOKENIZER =====
    model, tokenizer = load_or_download_model()
    tokenizer.pad_token = tokenizer.eos_token

    # ===== PREPARE MODEL =====
    model = prepare_model_for_kbit_training(model)
    
    # ===== LoRA CONFIG =====
    lora_config = LoraConfig(
        r=config['training_params']['lora_rank'],
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # ===== TOKENIZE DATA =====
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # ===== DATA COLLATOR =====
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ===== TRAINING ARGS =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name="tinyllama-egypt-qa",
        per_device_train_batch_size=config['training_params']['batch_size'],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        num_train_epochs=config['training_params']['num_epochs'],
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        group_by_length=True
    )

    # ===== TRAINER =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # ===== DEBUG INFO =====
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Trainable Parameters: {trainable_params:,}")
    logger.info(f"💻 Device Map: {model.hf_device_map}")

    # Verify GPU support
    if not torch.cuda.is_available():
        logger.warning("WARNING: Running on CPU - performance will be limited")
    else:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ===== TRAIN =====
    trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)

    # ===== SAVE =====
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    logger.info(f"🎯 Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
