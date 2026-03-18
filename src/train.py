from train_utils import train_model
import argparse
from dataset import BibleDataset
from model import CharacterTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--version", type=str, default="kjv", help="Bible version to train on")
    args = parser.parse_args()

    def progress(step, loss):
        print(f"[{args.version.upper()}] step {step}: loss {loss:.4f}")

    print(f"Starting CLI training session for version '{args.version}'...")
    train_model(max_iters=args.max_iters, progress_callback=progress, version=args.version)
    print(f"Training complete for version '{args.version}'.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
