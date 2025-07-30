import asyncio
import os
from data_pipeline import  extraction, cleaning, chunking, qa_generation, split_dataset, export_dataset,Agent
from training_evaluation_pipeline import training_pipeline,evaluation_pipeline
if __name__ == "__main__":
    print("🚀 Starting Data Pipeline...")


    folder_path = "data_set/raw"
    print("📄 Step 1: Extracting text...")
    text = extraction.run(folder_path)

   
    print("🧹 Step 2: Cleaning text...")
    tex_cleaned = cleaning.run(text)

    
    print("✂️ Step 3: Chunking text...")
    chunks = chunking.run(tex_cleaned)

    
    print("🤖 Step 4: Generating Agent...")
    agent = Agent.run()

    print("🤖 Step 5: Generating Q/A...")
    data_set =  asyncio.run(qa_generation.run(chunks[:1],agent))
    # 6️⃣ Split into Train/Val/Test
    print("📊 Step 6: Splitting dataset...")
    train_data, val_data, test_data = split_dataset.run(data_set)

    # 7️⃣ Export to JSONL for HF Format
    print("💾 Step 7: Exporting dataset...")
    export_dataset.run(train_data, val_data, test_data)

    print("✅ Data pipeline completed successfully.")

    print("Step 8: training pipeline...")
    training_pipeline.run()

    print("Step 9: evalution pipeline...")
    evaluation_pipeline.run()
