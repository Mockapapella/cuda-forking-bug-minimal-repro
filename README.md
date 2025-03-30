# tl;dr

This repository contains a minimal reproduction of the "Cannot re-initialize CUDA in forked subprocess." error that occurs when using CUDA with Gunicorn's forking model.

## The Issue

The issue occurs because CUDA cannot be shared across forked processes. When:

1. A parent process initializes CUDA (by importing torch in gunicorn.conf.py)
2. The process forks worker processes (Gunicorn creating worker processes)
3. Worker processes try to use CUDA

The worker processes fail with: "Cannot re-initialize CUDA in forked subprocess."

## How to Reproduce

1. Run the app directly (works fine):
```
python main.py
```

2. Run with gunicorn (will fail):
```
gunicorn -c gunicorn.conf.py main:app
```

## Root Cause

The CUDA runtime doesn't support the `fork()` system call. When a process that has initialized CUDA is forked, the child processes cannot initialize CUDA again, leading to errors when they try to use GPU resources.
