from train_utils import train_model
import argparse
from dataset import BibleDataset
from model import CharacterTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=None)
    args = parser.parse_args()

    def progress(step, loss):
        print(f"step {step}: loss {loss:.4f}")

    print("Starting CLI training session...")
    train_model(max_iters=args.max_iters, progress_callback=progress)
    print("Training complete.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
