"""Lightweight timing utility."""
import time
from contextlib import contextmanager

@contextmanager
def timer(block_name: str = "Code Block"):
    """Context manager that prints the wall-clock duration of a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{block_name}] executed in {elapsed:.4f} seconds")