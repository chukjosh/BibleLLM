import uvicorn
import argparse

def main():
    # Use argparse to allow some flexibility in starting the server
    parser = argparse.ArgumentParser(description="Run the Bible LLM FastAPI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind (localhost)")
    parser.add_argument("--port", type=int, default=8000, help="The port to run on")
    args = parser.parse_args()

    print(f"Starting Bible LLM API on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    
    # Run the uvicorn server pointing to our app in src/api.py
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
