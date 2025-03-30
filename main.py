"""Minimal FastAPI app that demonstrates CUDA fork issue."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
import torch
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    pid = os.getpid()
    print(f"Worker process {pid} initializing CUDA...")

    try:
        if torch.cuda.is_available():
            # This operation will fail with error when run with gunicorn
            # because the parent process already initialized CUDA
            tensor = torch.zeros(1, device="cuda:0")
            print(f"Worker process {pid} created tensor on device: {tensor.device}")
        else:
            print(f"Worker process {pid}: CUDA not available")

    except Exception as e:
        print(f"Worker process {pid} CUDA error: {e}")
        raise
    
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World", "pid": os.getpid()}


if __name__ == "__main__":
    print("Running directly (will work)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
