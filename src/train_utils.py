import torch
import torch.optim as optim
import os
from dataset import BibleDataset
from model import CharacterTransformer

# Hyperparameters for CPU-friendly training
DEFAULT_CONFIG = {
    "batch_size": 32,
    "block_size": 64,
    "max_iters": 5000,
    "eval_interval": 500,
    "learning_rate": 1e-3,
    "device": 'cpu',
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.2
}

def train_model(max_iters=None, progress_callback=None, version='kjv', resume=True):
    """
    Core training function that can be called from CLI or API.
    progress_callback: optional function(iter, loss) for reporting.
    resume: if True, tries to load existing model_{version}.pt to continue training.
    """
    config = DEFAULT_CONFIG.copy()
    if max_iters is not None:
        config["max_iters"] = max_iters

    # Load data
    version = version.lower()
    dataset_path = os.path.join('datasets', f'{version}.txt')
    dataset = BibleDataset(dataset_path, config["block_size"])
    
    # Model
    model = CharacterTransformer(
        vocab_size=dataset.vocab_size, 
        n_embd=config["n_embd"], 
        block_size=config["block_size"], 
        n_head=config["n_head"], 
        n_layer=config["n_layer"], 
        dropout=config["dropout"]
    )
    
    model_path = f'model_{version}.pt'
    if resume and os.path.exists(model_path):
        try:
            print(f"Resuming training from {model_path}...")
            checkpoint = torch.load(model_path, map_location=config["device"])
            # Ensure the checkpoint matches the model architecture
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Could not resume from checkpoint: {e}. Starting from scratch.")

    model.to(config["device"])
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    def get_batch():
        ix = torch.randint(len(dataset), (config["batch_size"],))
        x = torch.stack([dataset[i][0] for i in ix])
        y = torch.stack([dataset[i][1] for i in ix])
        return x.to(config["device"]), y.to(config["device"])

    last_loss = 0.0
    for iter in range(config["max_iters"]):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        last_loss = loss.item()
        
        if progress_callback and (iter % config["eval_interval"] == 0 or iter == config["max_iters"] - 1):
            progress_callback(iter, last_loss)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'n_embd': config["n_embd"],
        'block_size': config["block_size"],
        'n_head': config["n_head"],
        'n_layer': config["n_layer"]
    }, f'model_{version}.pt')
    
    return last_loss
