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
from model import CharacterTransformer
from train_utils import train_model
from lookup import lookup_verse

# Initialize FastAPI app
app = FastAPI(title="Bible LLM API", description="API for generating and training Biblical-style text")

# Define request/response models
class GenerateRequest(BaseModel):
    prompt: str = "In the beginning"
    max_tokens: int = 100
    version: str = "kjv"

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    version: str

class TrainRequest(BaseModel):
    max_iters: int = 500
    version: str = "kjv"

class StatusResponse(BaseModel):
    status: str
    last_iter: int
    last_loss: float
    version: str

# Global variables for model, mappings, and training state
current_model_version = None
model = None
stoi = None
itos = None
device = 'cpu'

# Training state tracking
training_state = {
    "status": "idle", # idle, running, complete, error
    "last_iter": 0,
    "last_loss": 0.0,
    "version": "kjv",
    "error": None
}

def load_model(version='kjv'):
    """Load the model and vocabulary for a specific version"""
    global model, stoi, itos, current_model_version
    version = version.lower()
    model_path = f'model_{version}.pt'
    
    # Check if we already have this model loaded
    if current_model_version == version and model is not None:
        return True

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
        current_model_version = version
        return True
    except Exception as e:
        print(f"Error loading model {version}: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Startup hook to load the default model (KJV)"""
    if load_model('kjv'):
        print("Default KJV model loaded successfully.")
    else:
        print("Default model (model_kjv.pt) not found. API available, but /generate will fail until training is complete.")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Bible LLM API. Multi-version support enabled.",
        "endpoints": ["POST /generate", "POST /train", "GET /status", "GET /versions"]
    }

@app.get("/versions")
async def get_versions():
    """List available Bible versions in the datasets folder"""
    if not os.path.exists('datasets'):
        return {"versions": []}
    files = [f.replace('.txt', '') for f in os.listdir('datasets') if f.endswith('.txt')]
    return {"versions": files}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Check the current training status"""
    return StatusResponse(
        status=training_state["status"],
        last_iter=training_state["last_iter"],
        last_loss=training_state["last_loss"],
        version=training_state["version"]
    )

def background_train(max_iters: int, version: str):
    """Background task for training a specific version"""
    global training_state
    try:
        training_state["status"] = "running"
        training_state["version"] = version
        training_state["error"] = None
        
        def update_progress(iter, loss):
            training_state["last_iter"] = iter
            training_state["last_loss"] = loss
        
        train_model(max_iters=max_iters, progress_callback=update_progress, version=version)
        
        # Reload model if it's the one currently active
        load_model(version)
        training_state["status"] = "complete"
    except Exception as e:
        training_state["status"] = "error"
        training_state["error"] = str(e)
        print(f"Training error ({version}): {e}")

@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger a training session in the background for a specific version"""
    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    
    # Check if dataset exists
    dataset_path = os.path.join('datasets', f'{request.version.lower()}.txt')
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset for version '{request.version}' not found.")

    background_tasks.add_task(background_train, request.max_iters, request.version)
    return {"message": f"Training started for version '{request.version}' in background."}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on a prompt and version"""
    
    # 1. Try exact lookup first for the requested version
    exact_match = lookup_verse(request.prompt, version=request.version)
    if exact_match:
        return GenerateResponse(prompt=request.prompt, generated_text=exact_match, version=request.version)

    # 2. Ensure the correct model version is loaded
    if current_model_version != request.version.lower():
        if not load_model(request.version):
            raise HTTPException(status_code=503, detail=f"Model for version '{request.version}' not loaded. Please train first.")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model error.")
    
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
