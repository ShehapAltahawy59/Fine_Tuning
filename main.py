import asyncio
import logging
import os

import yaml
from app.serving import APIServer
from data_pipeline import  extraction, cleaning, chunking, qa_generation, split_dataset, export_dataset,Agent
from training_evaluation_pipeline import training_pipeline,evaluation_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("🚀 Starting Data Pipeline...")
    logger.info("Starting Data Pipeline")

    folder_path = "data_set/raw"
    print("📄 Step 1: Extracting text...")
    logger.info("Step 1: Extracting text...")
    text = extraction.run(folder_path)

   
    print("🧹 Step 2: Cleaning text...")
    logger.info("Step 2: Cleaning text...")
    text_cleaned = cleaning.run(text)

    
    print("✂️ Step 3: Chunking text...")
    logger.info("Step 3: Chunking text...")
    chunks = chunking.run(text_cleaned)

    
    print("🤖 Step 4: Generating Agent...")
    logger.info("Step 4: Generating Agent...")
    agent = Agent.run()

    print("🤖 Step 5: Generating Q/A...")
    logger.info("Step 5: Generating Q/A...")
    data_set =  asyncio.run(qa_generation.run(chunks,agent))
    # 6️⃣ Split into Train/Val/Test
    print("📊 Step 6: Splitting dataset...")
    logger.info("Step 6: Splitting dataset...")
    train_data, val_data, test_data = split_dataset.run(data_set)

    # 7️⃣ Export to JSONL for HF Format
    print("💾 Step 7: Exporting dataset...")
    logger.info("Step 7: Exporting dataset...")
    export_dataset.run(train_data, val_data, test_data)

    logger.info("Data pipeline completed successfully.")
    print("✅ Data pipeline completed successfully.")

    print("Step 8: training pipeline...")
    # logger.info("Step 8: training pipeline...")
    # training_pipeline.run()

    # print(" training completed")
    # logger.info("training completed")

    # print("Step 9: evalution pipeline...")
    # logger.info("Step 9: evalution pipeline...")
    # evaluation_pipeline.run()

    # print(" evalution completed")
    # logger.info("evalution completed")

    from app import serving
    logger.info("step 10: Starting deployment")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize and run
    server = APIServer(config)
    server.run()
