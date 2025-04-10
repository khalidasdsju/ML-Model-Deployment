#!/usr/bin/env python3
"""
Main entry point for the Heart Failure Prediction API.
This file imports and runs the FastAPI application from src/api/app.py.
"""
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app from the api module
from src.api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
