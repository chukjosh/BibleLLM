# Bible LLM

A lightweight, PyTorch-based character-level Transformer Language Model trained on the King James Version of the Bible (`datasets/kjv.txt`). Optimized for CPU training.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the model on `datasets/kjv.txt`, run:

```bash
python src/train.py
```

The training script will save the trained model weights to `model.pt` in the project root.

## Generation

To generate text using the trained model, run:

```bash
python src/generate.py
```
