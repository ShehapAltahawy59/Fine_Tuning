
# 📜 Egypt History Data Pipeline (Dev Branch)

## 📌 Overview
This branch contains the **data pipeline** for preparing a domain-specific dataset on **Egyptian History**.  
The pipeline:
1. Ingests data (PDFs, web sources)
2. Extracts and cleans text
3. Generates domain-specific Q&A pairs automatically using **AutoGen agents**
4. Structures the dataset into **train/val/test** for fine-tuning (Open-Book & Closed-Book)

---

## 🏛 Why Egyptian History?
We chose **Egyptian History** because:
- Rich, factual, and well-documented domain
- High potential for **Closed-Book** fine-tuning (facts need memorization)
- **Open-Book** also benefits from context due to long historical documents
- Availability of **public domain sources** (books, encyclopedias, history archives)

---

## 📂 Pipeline Steps
### **1️⃣ Data Ingestion**
- PDFs of Egyptian History are downloaded from public sources
- Saved in `datasets/raw/`

### **2️⃣ Text Extraction (PyMuPDF)**
- We use `PyMuPDF (fitz)` to extract **block-level text** from PDFs
- Text is cleaned of headers, footers, artifacts

Example:
```python
import fitz
doc = fitz.open("history.pdf")
text = "\n".join(page.get_text() for page in doc)
````

### **3️⃣ Text Cleaning & Chunking**

* Remove page numbers, broken lines
* Split into **chunks** for Q\&A generation using `RecursiveCharacterTextSplitter` (LangChain)

---

## 🤖 AutoGen Q\&A Generation

We use **AutoGen agents** to:

* Take each text chunk
* Generate **multiple question-answer pairs** in JSON format
* Store in structured dataset

Example agent workflow:

```python
from autogen_agentchat.agents import AssistantAgent
agent = AssistantAgent("EgyptHistoryQA")

qas = agent.generate_qa(context_chunk)
```

---

## 📊 Dataset Structure

The final dataset is saved as:

```json
{
  "context": "...",
  "question": "...",
  "answer": "..."
}
```

* **Train set** → Used for fine-tuning
* **Validation set** → Used during training evaluation
* **Test set** → Used for benchmark comparison (baseline vs fine-tuned)

---

## 🚀 How to Run the Pipeline

### **1️⃣ Install Requirements**

```bash
pip install -r requirements.txt
```

### **2️⃣ Set API Keys (AutoGen, OpenAI, Gemini, etc.)**

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

### **3️⃣ Run the Pipeline**

```bash
python main.py
```

This will:

* Ingest PDFs
* Extract & clean text
* Generate Q\&A
* Export `train.jsonl`, `val.jsonl`, `test.jsonl` in `datasets/final/`

---

## ✅ Next Steps

Once dataset is ready:

* Merge `dev` → `training-openbook`
* Merge `dev` → `training-closedbook`

---

## 📌 Notes

* All data used is from **public domain** or allowed under fair use
* AutoGen agents are configured to produce **factual, concise questions**

