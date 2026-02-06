"""
AgenticAffect: Emotion Classification Multi-Agent System
Entry point for the multi-agent pipeline.
"""

import os
import sys
import ctypes

# WORKAROUND: Pre-load libiomp5md.dll to fix PyTorch initialization error on Windows
# This must be done before importing torch or libraries that use torch (like datasets/transformers)
try:
    site_packages = os.path.join(
        os.path.dirname(sys.executable), "Lib", "site-packages"
    )
    torch_lib = os.path.join(site_packages, "torch", "lib")
    omp_path = os.path.join(torch_lib, "libiomp5md.dll")

    if os.path.exists(omp_path):
        ctypes.CDLL(omp_path)
except Exception:
    pass

# Import torch BEFORE datasets/other libs to ensure the correct OpenMP is loaded
import torch
import time
import warnings

# Suppress symlink warnings on Windows
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from datasets import load_dataset
import pandas as pd
from tasks.pipeline import Pipeline


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting AgenticAffect system...")

    try:
        # Load dataset
        print(f"[{time.strftime('%H:%M:%S')}] Loading 'emotion' dataset...")
        dataset = load_dataset("emotion")
        train = dataset["train"].to_pandas().sample(10, random_state=42)
        texts = train["text"].tolist()
        labels = train["label"].tolist()
        print(
            f"[{time.strftime('%H:%M:%S')}] Dataset loaded successfully ({len(texts)} samples)."
        )

        # Set up the pipeline with texts and labels
        print(f"[{time.strftime('%H:%M:%S')}] Initializing pipeline components...")
        print(
            f"[{time.strftime('%H:%M:%S')}] Note: This may take a few minutes if downloading the model (260MB) for the first time."
        )
        pipeline = Pipeline(texts, labels)

        print(
            f"[{time.strftime('%H:%M:%S')}] Pipeline initialized. Running analysis..."
        )
        pipeline.run()
        print(f"[{time.strftime('%H:%M:%S')}] Execution completed successfully.")

    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease share the above error message if asked.")


if __name__ == "__main__":
    main()
