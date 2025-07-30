# training_pipeline.py
import logging
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from transformers import TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training
import torch
import wandb
import os
from dotenv import load_dotenv
from huggingface_hub import login
import yaml
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
huggingface_key = os.environ["huggingface_key"]
login(huggingface_key)
wandb_key = os.environ["wandb_key"]
wandb.login(key=wandb_key)
MODEL_NAME = config['base_model']
OUTPUT_DIR = config["output_dir"]
RESUME_CHECKPOINT = None  # set path to resume, or None for fresh start
MAX_LENGTH = 512
def run():
    # ===== LOAD DATASET =====
    dataset = load_dataset("json", data_files={
        "train": "./data_set/final/egypt_pdf_qa_train.jsonl",
        "validation": "./data_set/final/egypt_pdf_qa_val.jsonl",
        "test": "./data_set/final/egypt_pdf_qa_test.jsonl"
    })

    # ===== FORMAT FUNCTIONS =====
    def format_propmet(example):
        return {
            "instruction": "Answer the question concisely.",
            "input": example['question'],
            "output": example["answer"]
        }


    def tokenize_fn(example):
        formatted = format_propmet(example)
        
        # Combine instruction + input into a single text prompt
        prompt = f"{formatted['instruction']}\n{formatted['input']}"
        
        return tokenizer(
            prompt,
            text_target=formatted["output"],   
            padding="max_length",
            truncation=True,
            max_length=512
        )

    # ===== LOAD TOKENIZER & MODEL =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        token=True
    )

    # ===== LoRA CONFIG =====
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config['training_params']['lora_rank'],
        lora_alpha=16,
        target_modules=["q_proj", "v_proj","k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    # ===== TOKENIZE DATA =====
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # ===== TRAINING ARGS =====
    training_args = TrainingArguments(
        output_dir="./mistral_qlora_egypt_closed",
        report_to="wandb",              #  enable W&B tracking
        run_name="mistral_qlora_egypt_closed",
        per_device_train_batch_size=config['training_params']['batch_size'],   
        per_device_eval_batch_size=1,    
        gradient_accumulation_steps=8,   # effective batch size
        eval_strategy="epoch",
        label_names=["labels"],
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=config['training_params']['num_epochs'],
        learning_rate=config['training_params']['learning_rate'],
        fp16=True,
        gradient_checkpointing=True,     
        logging_dir="./logs",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )


    # ===== DEBUG INFO =====
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Trainable Parameters: {trainable_params}")

    # ===== TRAIN =====
    if RESUME_CHECKPOINT:
        trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
    else:
        trainer.train()

    # ===== SAVE =====
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

    logger.info("🎯 Training Complete. Model saved.")
