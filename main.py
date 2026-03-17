"""
Bible LLM API Entry Point
-------------------------
This script serves as the main entry point for the Bible LLM FastAPI inference server.
It uses 'uvicorn' to host the FastAPI application defined in 'src/api.py'.
"""

import uvicorn
import argparse

def main():
    # Configure command-line arguments to allow host/port flexibility
    parser = argparse.ArgumentParser(description="Run the Bible LLM FastAPI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="The host IP address to bind (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="The port number to run on (default: 8000)")
    args = parser.parse_args()

    # Inform the user about the server status
    print(f"Starting Bible LLM API on http://{args.host}:{args.port}")
    print("Interactive Swagger UI available at http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop.")
    
    # Run the uvicorn server. 
    # The 'src.api:app' string points to the 'app' object inside 'src/api.py'.
    # reload=False is safer for production; set to True during development to auto-restart on code changes.
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
