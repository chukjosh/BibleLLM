import os
import torch
from torch.utils.data import Dataset

class BibleDataset(Dataset):
    """
    Custom Dataset for character-level language modeling.
    Reads a text file and converts it into a sequence of integers.
    """
    def __init__(self, file_path, block_size):
        super().__init__()
        # Load the raw text from the bible file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        
        # Build vocabulary: get all unique characters and sort them
        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        
        # Create mapping from character to integer (stoi) and vice versa (itos)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.block_size = block_size
        
        # Pre-encode the entire dataset into a tensor of integers.
        # This saves time during training as we don't have to encode on-the-fly.
        self.encoded_data = torch.tensor([self.stoi[c] for c in self.data], dtype=torch.long)

    def __len__(self):
        # We can start a block at any index that leaves enough room for block_size + 1 (for target)
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of indices: the context [x] and the next character [y]
        # x is the sequence of length block_size
        # y is the same sequence shifted by one (the target for every position in x)
        chunk = self.encoded_data[idx:idx + self.block_size + 1]
        
        x = chunk[:-1] # input sequence (B, T)
        y = chunk[1:]  # target sequence (B, T)
        
        return x, y

    def encode(self, s):
        """Helper to convert a string of characters into a list of integers."""
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        """Helper to convert a list of integers back into a human-readable string."""
        return ''.join([self.itos[i] for i in l])
