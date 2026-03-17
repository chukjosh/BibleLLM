import os
import torch
from torch.utils.data import Dataset

class BibleDataset(Dataset):
    def __init__(self, file_path, block_size):
        super().__init__()
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        
        # Build vocabulary
        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.block_size = block_size
        
        # We encode the full dataset here as it easily fits in RAM for kjv.txt 
        # (approx 4.6 MB)
        self.encoded_data = torch.tensor([self.stoi[c] for c in self.data], dtype=torch.long)

    def __len__(self):
        # We can extract a block from anywhere up to the end minus block_size
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        # We grab a chunk of (block_size + 1) characters from the data
        chunk = self.encoded_data[idx:idx + self.block_size + 1]
        
        # Inputs to the transformer
        x = chunk[:-1]
        
        # Targets for the transformer (shifted by 1)
        y = chunk[1:]
        
        return x, y

    def encode(self, s):
        """Encode a string into a list of integers"""
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        """Decode a list of integers back to a string"""
        return ''.join([self.itos[i] for i in l])
