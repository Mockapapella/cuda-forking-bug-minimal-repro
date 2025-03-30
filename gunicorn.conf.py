"""Gunicorn configuration file."""

import os
import torch


# Device detection logic similar to what was in the postmortem
if torch.backends.mps.is_available():
    DEVICE_TYPE = "MPS"
elif torch.cuda.is_available():
    DEVICE_TYPE = "GPU"
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE_TYPE = "CPU"

print(f"Parent process {os.getpid()} using device: {DEVICE_TYPE}")

# Standard gunicorn settings
bind = "0.0.0.0:8000"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"

print(f"Gunicorn loaded config in parent process, will fork {workers} workers")
