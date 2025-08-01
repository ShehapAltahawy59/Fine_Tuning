from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from . import API_KEY,tokenizer,model,MAX_NEW_TOKENS


# ===== FASTAPI APP =====
app = FastAPI(title="Egypt History QA API", version="1.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    latency: float

@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    try:
        start_time = time.time()
        prompt = f"Question: {request.question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        latency = time.time() - start_time
        
        return QueryResponse(answer=answer, latency=latency)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Egypt History QA API is running!"}


class APIServer:
    def __init__(self, config):
        self.app = app
        self.config = config
    
    def run(self):
        import uvicorn
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.config['deployment']['port']
        )
# def run(self):
#         import uvicorn
#         uvicorn.run(
#             self.app,
#             host="0.0.0.0",
#             port=self.config['deployment']['port']
#         )
