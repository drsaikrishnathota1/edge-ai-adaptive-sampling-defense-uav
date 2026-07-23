#!/usr/bin/env python3
"""Download and extract the official VisDrone2019-VID validation set."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dataset_cfg = cfg["dataset"]
    zip_path = Path(dataset_cfg["zip_path"])
    extract_dir = Path(dataset_cfg["extract_dir"])
    file_id = dataset_cfg["google_drive_file_id"]

    if extract_dir.exists() and any(extract_dir.rglob("*.jpg")):
        print(f"Dataset already extracted at {extract_dir}")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading dataset to {zip_path}")
        output = gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
        if output is None or not zip_path.exists():
            raise RuntimeError(
                "Dataset download failed. Re-run the command or download the "
                "official VisDrone2019-VID-val.zip into data/ manually."
            )

    print(f"Extracting {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    print(f"Dataset extracted at {extract_dir}")


if __name__ == "__main__":
    main()
