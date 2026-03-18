"""
Bible LLM API Entry Point
-------------------------
This script serves as the main entry point for the Bible LLM FastAPI inference server.
It uses 'uvicorn' to host the FastAPI application defined in 'src/api.py'.
"""

import uvicorn
import argparse
import sys
import os

# Add the 'src' directory to the path so that imports within api.py work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
    # Since 'src' is in sys.path, we can just use 'api:app'.
    uvicorn.run("api:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
