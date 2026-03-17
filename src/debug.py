import torch
from dataset import BibleDataset
from model import CharacterTransformer
import os

print("Modules imported successfully!")
dataset_path = os.path.join('datasets', 'kjv.txt')
if os.path.exists(dataset_path):
    print(f"Dataset found at {dataset_path}")
    dataset = BibleDataset(dataset_path, 64)
    print(f"Vocab size: {dataset.vocab_size}")
else:
    print(f"Dataset NOT found at {dataset_path}")

model = CharacterTransformer(vocab_size=100, n_embd=128, n_head=4, n_layer=4)
print("Model initialized successfully!")
