import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        # Linear layers for Key, Query, and Value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Triangular mask for causal (autoregressive) attention
        # register_buffer ensures it's not considered a parameter but is moved with the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # Scale by 1/sqrt(dk) to prevent gradients from exploding
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        
        # Mask out future tokens ("causal mask") so the model can only look at the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax normalize scores to get weight distribution
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v    # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention executing in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        # Initialize multiple attention heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        # Final projection to bring concatenated heads back to n_embd dimension
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply final linear projection and dropout
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity (ReLU) and another linear layer """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Standard expansion to 4x dimension
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection back to original dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ 
    Transformer block: Communication (Self-Attention) followed by Computation (Feed-Forward).
    Uses residual connections and layer normalization.
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        # LayerNorm layers are applied BEFORE the attention and feed-forward modules
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Add residual connections to help gradient flow during training
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CharacterTransformer(nn.Module):
    """ The main Transformer Language Model class """
    def __init__(self, vocab_size, n_embd=256, block_size=256, n_head=4, n_layer=4, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Token embedding: map characters to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position embedding: represent the location of characters in the sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Stack multiple Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        
        # Final layer norm before the classification head
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Classification head: map embeddings to character logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of character indices
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        
        # Combine token and position information
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Pass through the Transformer blocks
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x)   # (B, T, n_embd)
        
        # Compute logits for the next character in the sequence
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
            return logits, loss
        else:
            # Flatten tensors to calculate CrossEntropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Autoregressive character generation """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens to fit in positional embeddings
            idx_cond = idx[:, -self.block_size:]
            # Get the predictions for the next token
            logits, _ = self(idx_cond)
            # Focus only on the last time step (B, T, C) -> (B, C)
            logits = logits[:, -1, :] 
            # Apply softmax to get character probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
