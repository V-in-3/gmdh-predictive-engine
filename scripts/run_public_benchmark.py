import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark evaluation pipeline on a public dataset")
    parser.add_argument("url", help="URL to a public CSV benchmark")
    parser.add_argument("--output_dir", default="data", help="Directory where outputs will be written")
    parser.add_argument("--source_type", default="ieee_cis", choices=["ieee_cis"], help="Benchmark source")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = output_dir / "benchmark_normalized.csv"
    model_path = output_dir / "benchmark_model.json"
    metrics_path = output_dir / "benchmark_metrics.json"

    command = [
        sys.executable,
        "scripts/benchmark_eval.py",
        args.url,
        "--output_path",
        str(normalized_path),
        "--model_output_path",
        str(model_path),
        "--metrics_output_path",
        str(metrics_path),
        "--source_type",
        args.source_type,
    ]

    print("Running benchmark evaluation...")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print("Completed.")
    print(f"Normalized CSV: {normalized_path}")
    print(f"Model JSON: {model_path}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
