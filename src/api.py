from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
from .model import CharacterTransformer

# Initialize FastAPI app
app = FastAPI(title="Bible LLM API", description="API for generating Biblical-style text")

# Define request/response models
class GenerateRequest(BaseModel):
    prompt: str = "In the beginning"
    max_tokens: int = 100

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str

# Global variables for model and mappings
model = None
stoi = None
itos = None
device = 'cpu'

def load_model():
    """Load the model and vocabulary from model.pt"""
    global model, stoi, itos
    model_path = 'model.pt'
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file '{model_path}' not found. Please train the model first.")
        
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct the model
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
    
    # Load mappings
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

@app.on_event("startup")
async def startup_event():
    """Startup hook to load the model once"""
    try:
        load_model()
        print("Model loaded successfully for API.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Bible LLM API. Use /docs for Swagger UI or POST to /generate."}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on a prompt"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or training in progress.")
    
    try:
        # Encoding/Decoding helpers
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: ''.join([itos.get(i, '?') for i in l])
        
        # Prepare context
        context = torch.tensor([encode(request.prompt)], dtype=torch.long, device=device)
        
        # Generate tokens
        with torch.no_grad():
            generated_indices = model.generate(context, max_new_tokens=request.max_tokens)
        
        # Decode and return
        res_text = decode(generated_indices[0].tolist())
        return GenerateResponse(prompt=request.prompt, generated_text=res_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
