# Bible LLM

A lightweight, PyTorch-based character-level Transformer Language Model designed for Biblical-style text generation. Optimized for CPU training and featuring a hybrid search capability for exact verse retrieval.

## Features

- **Hybrid Search**: Automatically detects Bible references (e.g., "Genesis 1:1") and returns exact text from the dataset before falling back to AI generation.
- **Multi-Version Support**: Specify different Bible versions (KJV, WEB, etc.) for training and generation.
- **Resumable Training**: Smart checkpoints allow you to pause and resume training sessions without losing progress.
- **FastAPI Integration**: A fully functional REST API for remote training and text generation.

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

## Datasets

Place your Bible text files in the `datasets/` directory. Files must be named `{version}.txt` (e.g., `kjv.txt`, `web.txt`).

## Usage

### 1. Training (via CLI)

To train the model on a specific version:
```bash
python src/train.py --version kjv --max_iters 5000
```
- Use `--no-resume` to start training from scratch instead of continuing from a previous session.
- Weights are saved to `model_{version}.pt`.

### 2. Generation (via CLI)

Generate text or look up a verse:
```bash
python src/generate.py --version kjv --prompt "Genesis 1:1"
```
- If the prompt is a citation, it returns the exact verse.
- Otherwise, it uses the AI model to generate text.

### 3. API (FastAPI)

Run the server:
```bash
python main.py
```

**Generate Text**:
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "In the beginning", "version": "kjv", "max_tokens": 100}'
```

**Train Model**:
```bash
curl -X POST "http://127.0.0.1:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"version": "kjv", "max_iters": 500, "resume": true}'
```

**Utility Endpoints**:
- `GET /versions`: List available datasets.
- `GET /status`: Check current training progress.
- `http://127.0.0.1:8000/docs`: Interactive Swagger UI.
