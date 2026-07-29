#!/usr/bin/env python3
"""
Fine-tune TinyLlama for fraud analysis using MLX on Apple Silicon.

Trains a LoRA adapter on fraud case analysis examples.
Result can be exported to Ollama via GGUF conversion.

Requirements:
    pip install mlx-lm

Usage:
    python scripts/finetune_fraud_llm.py

Output:
    models/fraud-analyst-lora/  — LoRA adapter weights
"""
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "fraud_training_llm.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "fraud-analyst-lora"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training hyperparameters
TRAIN_CONFIG = {
    "model": MODEL_NAME,
    "train": True,
    "data": str(DATA_FILE.parent),
    "iters": 200,             # training iterations (small dataset = fewer iters)
    "batch_size": 1,          # M1 Pro 16GB safe
    "lora_layers": 8,         # number of LoRA layers
    "learning_rate": 1e-4,
    "adapter_path": str(OUTPUT_DIR),
}


def prepare_data():
    """Split training data into train/valid/test for MLX format."""
    with open(DATA_FILE) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Total examples: {len(examples)}")

    # MLX expects train.jsonl, valid.jsonl, test.jsonl in data dir
    train_file = DATA_FILE.parent / "train.jsonl"
    valid_file = DATA_FILE.parent / "valid.jsonl"
    test_file = DATA_FILE.parent / "test.jsonl"

    # 80% train, 10% valid, 10% test
    n = len(examples)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    with open(train_file, "w") as f:
        for ex in examples[:train_end]:
            f.write(json.dumps(ex) + "\n")

    with open(valid_file, "w") as f:
        for ex in examples[train_end:valid_end]:
            f.write(json.dumps(ex) + "\n")

    with open(test_file, "w") as f:
        for ex in examples[valid_end:]:
            f.write(json.dumps(ex) + "\n")

    print(f"Split: {train_end} train / {valid_end - train_end} valid / {n - valid_end} test")
    return train_file.parent


def run_finetune():
    """Run LoRA fine-tuning via MLX."""
    from mlx_lm import lora

    data_dir = prepare_data()

    print(f"\n{'=' * 70}")
    print(f"FINE-TUNING: {MODEL_NAME}")
    print(f"{'=' * 70}")
    print(f"  Data:       {data_dir}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Iterations: {TRAIN_CONFIG['iters']}")
    print(f"  LoRA layers: {TRAIN_CONFIG['lora_layers']}")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"{'=' * 70}\n")

    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run training
    lora.train(
        model=MODEL_NAME,
        train=True,
        data=str(data_dir),
        iters=TRAIN_CONFIG["iters"],
        batch_size=TRAIN_CONFIG["batch_size"],
        lora_layers=TRAIN_CONFIG["lora_layers"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        adapter_path=str(OUTPUT_DIR),
    )

    print(f"\n{'=' * 70}")
    print(f"DONE! LoRA adapter saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    print(f"\nNext steps:")
    print(f"  1. Test: python -c \"from mlx_lm import load, generate; ...\"")
    print(f"  2. Fuse:  mlx_lm.fuse --model {MODEL_NAME} --adapter-path {OUTPUT_DIR}")
    print(f"  3. Convert to GGUF for Ollama")


def test_model():
    """Quick test of the fine-tuned model."""
    from mlx_lm import load, generate

    print("\nTesting fine-tuned model...")
    model, tokenizer = load(MODEL_NAME, adapter_path=str(OUTPUT_DIR))

    prompt = """<|user|>
You are a fraud analyst. Analyze this transaction.

TRANSACTION:
- Semantic risk: 0.75
- Velocity: 15/hr
- Proxy score: 0.6
- Amount deviation: 1.8

SIMILAR CASES:
  Case #1 (FRAUD, dist=0.003): Bulk purchase smartphones, VPN, new account

Provide: BLOCK/ALLOW/REVIEW + reasoning
<|assistant|>
"""

    response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(f"\nModel response:\n{response}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_model()
    else:
        run_finetune()
        print("\n\nRun with --test to verify the model:")
        print(f"  python {__file__} --test")
