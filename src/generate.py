from model import CharacterTransformer
import os
import argparse
from lookup import lookup_verse

# Same hyperparameters as training (loaded from model.pt)
device = 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="In the beginning", help="The start of the text generation")
    parser.add_argument("--max_tokens", type=int, default=500, help="How many characters to generate")
    args = parser.parse_args()

    # 1. Try exact lookup first
    exact_match = lookup_verse(args.prompt)
    if exact_match:
        print("-" * 30)
        print(f"Exact Verse Found: '{args.prompt}'")
        print("-" * 30)
        print(exact_match)
        return

    # 2. Fall back to LLM generation
    # Load the comprehensive model checkpoint
    model_path = 'model.pt'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please run 'python src/train.py' first.")
        return

    # Load weights and hyperparameters from the .pt file
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct the exact model architecture used during training
    model = CharacterTransformer(
        vocab_size=checkpoint['vocab_size'],
        n_embd=checkpoint['n_embd'],
        block_size=checkpoint['block_size'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Set to evaluation mode (disables dropout)

    # Reconstruct character mappings
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Lambda functions for quick encoding/decoding
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])

    # Convert prompt to the starting context tensor
    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    
    print("-" * 30)
    print(f"Generating from: '{args.prompt}'")
    print("-" * 30)
    
    # Use the model's iterative generate function to produce characters
    generated = model.generate(context, max_new_tokens=args.max_tokens)
    
    # Print the resulting complete text
    print(decode(generated[0].tolist()))

if __name__ == "__main__":
    main()
