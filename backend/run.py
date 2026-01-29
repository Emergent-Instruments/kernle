#!/usr/bin/env python3
"""Entry point for Railway deployment."""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Kernle Backend on port {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
