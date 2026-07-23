#!/usr/bin/env python3
"""Run the complete reproducible experiment with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an already extracted VisDrone dataset.",
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if not args.skip_download:
        run([sys.executable, "src/download_dataset.py", "--config", args.config])

    run([sys.executable, "src/prepare_features.py", "--config", args.config])
    run([sys.executable, "src/train_and_evaluate.py", "--config", args.config])
    run([sys.executable, "src/create_figures.py", "--config", args.config])

    print("\nExperiment completed.")
    print("Results: results/")
    print("Figures: figures/")
    print("Model: models/adaptive_sampling_rf.joblib")


if __name__ == "__main__":
    main()
