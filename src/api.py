"""
Bible LLM FastAPI Implementation
--------------------------------
This module defines the REST API for the Bible LLM project.
It handles model loading on startup and provides an endpoint for text generation.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import os
import threading
from .model import CharacterTransformer
from .train_utils import train_model

# Initialize FastAPI app
app = FastAPI(title="Bible LLM API", description="API for generating and training Biblical-style text")

# Define request/response models
class GenerateRequest(BaseModel):
    prompt: str = "In the beginning"
    max_tokens: int = 100

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str

class TrainRequest(BaseModel):
    max_iters: int = 500

class StatusResponse(BaseModel):
    status: str
    last_iter: int
    last_loss: float

# Global variables for model, mappings, and training state
model = None
stoi = None
itos = None
device = 'cpu'

# Training state tracking
training_state = {
    "status": "idle", # idle, running, complete, error
    "last_iter": 0,
    "last_loss": 0.0,
    "error": None
}

def load_model():
    """Load the model and vocabulary from model.pt"""
    global model, stoi, itos
    model_path = 'model.pt'
    
    if not os.path.exists(model_path):
        return False
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = CharacterTransformer(
            vocab_size=checkpoint['vocab_size'],
            n_embd=checkpoint['n_embd'],
            block_size=checkpoint['block_size'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        stoi = checkpoint['stoi']
        itos = checkpoint['itos']
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Startup hook to load the model once"""
    if load_model():
        print("Model loaded successfully for API.")
    else:
        print("Model file not found or failed to load. API available, but /generate will fail until training is complete.")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Bible LLM API. Use /docs for Swagger UI.",
        "endpoints": ["POST /generate", "POST /train", "GET /status"]
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Check the current training status"""
    return StatusResponse(
        status=training_state["status"],
        last_iter=training_state["last_iter"],
        last_loss=training_state["last_loss"]
    )

def background_train(max_iters: int):
    """Background task for training"""
    global training_state
    try:
        training_state["status"] = "running"
        training_state["error"] = None
        
        def update_progress(iter, loss):
            training_state["last_iter"] = iter
            training_state["last_loss"] = loss
        
        train_model(max_iters=max_iters, progress_callback=update_progress)
        
        # Reload model after training completes
        load_model()
        training_state["status"] = "complete"
    except Exception as e:
        training_state["status"] = "error"
        training_state["error"] = str(e)
        print(f"Training error: {e}")

@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger a training session in the background"""
    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    
    background_tasks.add_task(background_train, request.max_iters)
    return {"message": "Training started in background."}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on a prompt"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train or check /status.")
    
    try:
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: ''.join([itos.get(i, '?') for i in l])
        context = torch.tensor([encode(request.prompt)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated_indices = model.generate(context, max_new_tokens=request.max_tokens)
        
        res_text = decode(generated_indices[0].tolist())
        return GenerateResponse(prompt=request.prompt, generated_text=res_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
