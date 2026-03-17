import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BibleDataset
from model import CharacterTransformer
from tqdm import tqdm
import os
import argparse

# Hyperparameters for CPU-friendly training
batch_size = 32
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

def main():
    parser = argparse.ArgumentParser()
    # Allow overiding max iterations via CLI
    parser.add_argument("--max_iters", type=int, default=max_iters)
    args = parser.parse_args()

    # Initialize Dataset
    dataset_path = os.path.join('datasets', 'kjv.txt')
    dataset = BibleDataset(dataset_path, block_size)
    
    # Initialize Model with CPU-optimized hyperparameters
    model = CharacterTransformer(
        vocab_size=dataset.vocab_size, 
        n_embd=n_embd, 
        block_size=block_size, 
        n_head=n_head, 
        n_layer=n_layer, 
        dropout=dropout
    )
    model.to(device)
    
    # Use AdamW optimizer: Decoupled Weight Decay Regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {args.max_iters} iterations on {device}...")
    
    # Helper to fetch a random batch of data
    def get_batch():
        # Generate random start indices for each sample in the batch
        ix = torch.randint(len(dataset), (batch_size,))
        # Stack the input-target pairs into tensors
        x = torch.stack([dataset[i][0] for i in ix])
        y = torch.stack([dataset[i][1] for i in ix])
        return x.to(device), y.to(device)

    # Standard training loop
    for iter in range(args.max_iters):
        # Get next batch
        xb, yb = get_batch()

        # Forward pass: compute logits and cross-entropy loss
        logits, loss = model(xb, yb)
        
        # Backward pass: compute gradients and update weights
        optimizer.zero_grad(set_to_none=True) # set_to_none=True is slightly more efficient
        loss.backward()
        optimizer.step()

        # Report training progress
        if iter % eval_interval == 0 or iter == args.max_iters - 1:
            print(f"step {iter}: loss {loss.item():.4f}")

    # Save the model state and required metadata (vocabulary and architecture info)
    # This ensures everything needed for generation is in one file.
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'n_embd': n_embd,
        'block_size': block_size,
        'n_head': n_head,
        'n_layer': n_layer
    }, 'model.pt')
    print("Model saved to model.pt")

if __name__ == "__main__":
    main()
