#!/usr/bin/env python3
"""Create lightweight visual features and annotation-derived event labels."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


VISDRONE_COLUMNS = [
    "frame",
    "target_id",
    "x",
    "y",
    "width",
    "height",
    "score",
    "category",
    "truncation",
    "occlusion",
]


def locate_dataset_root(extract_dir: Path) -> Path:
    candidates = [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if (candidate / "sequences").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find 'sequences' and 'annotations' under {extract_dir}."
    )


def read_sequence_fps(sequence_dir: Path, default_fps: int) -> int:
    possible = [
        sequence_dir / "seqinfo.ini",
        sequence_dir.parent.parent / "sequenceInfo" / f"{sequence_dir.name}.ini",
    ]
    for path in possible:
        if not path.exists():
            continue
        parser = configparser.ConfigParser()
        parser.read(path)
        for section in parser.sections():
            for key in ("frameRate", "framerate", "fps"):
                if parser.has_option(section, key):
                    try:
                        return int(float(parser.get(section, key)))
                    except ValueError:
                        pass
    return default_fps


def read_annotations(path: Path, target_categories: set[int]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=VISDRONE_COLUMNS)

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 10:
        raise ValueError(f"Unexpected annotation format in {path}")
    df = df.iloc[:, :10]
    df.columns = VISDRONE_COLUMNS

    valid = (df["score"] != 0) & df["category"].isin(target_categories)
    return df.loc[valid].copy()


def frame_feature_vector(
    gray: np.ndarray,
    previous_gray: np.ndarray | None,
) -> dict[str, float]:
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    mean_intensity = float(np.mean(small))
    std_intensity = float(np.std(small))

    histogram = cv2.calcHist([small], [0], None, [32], [0, 256]).ravel()
    histogram = histogram / max(float(histogram.sum()), 1.0)
    positive = histogram[histogram > 0]
    entropy = float(-(positive * np.log2(positive)).sum())

    edges = cv2.Canny(small, 80, 160)
    edge_density = float(np.mean(edges > 0))

    motion_score = 0.0
    if previous_gray is not None:
        previous_small = cv2.resize(
            previous_gray, (160, 90), interpolation=cv2.INTER_AREA
        )
        motion_score = float(cv2.absdiff(small, previous_small).mean())

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "entropy": entropy,
        "edge_density": edge_density,
        "motion_score": motion_score,
    }


def oracle_required_fps(
    current_objects: int,
    next_objects: int,
    motion_score: float,
) -> int:
    """Define the minimum desirable rate from future activity and scene change.

    This label is used only for supervised training and is never available to
    the controller during testing.
    """
    change = abs(next_objects - current_objects)

    if current_objects == 0 and next_objects == 0 and motion_score < 6:
        return 1
    if max(current_objects, next_objects) <= 2 and change == 0 and motion_score < 12:
        return 5
    if max(current_objects, next_objects) <= 6 and change <= 2 and motion_score < 22:
        return 10
    return 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dataset_cfg = cfg["dataset"]
    experiment_cfg = cfg["experiment"]

    extract_dir = Path(dataset_cfg["extract_dir"])
    root = locate_dataset_root(extract_dir)
    sequences_dir = root / "sequences"
    annotations_dir = root / "annotations"

    sequence_dirs = sorted([p for p in sequences_dir.iterdir() if p.is_dir()])
    max_sequences = int(dataset_cfg["max_sequences"])
    if max_sequences > 0:
        sequence_dirs = sequence_dirs[:max_sequences]

    if not sequence_dirs:
        raise RuntimeError("No sequence directories were found.")

    target_categories = set(map(int, experiment_cfg["target_categories"]))
    max_frames = int(dataset_cfg["max_frames_per_sequence"])
    default_fps = int(dataset_cfg["default_source_fps"])

    rows: list[dict] = []

    for sequence_dir in sequence_dirs:
        frame_paths = sorted(sequence_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(sequence_dir.glob("*.png"))
        if max_frames > 0:
            frame_paths = frame_paths[:max_frames]

        annotation_path = annotations_dir / f"{sequence_dir.name}.txt"
        annotations = read_annotations(annotation_path, target_categories)
        counts = annotations.groupby("frame").size().to_dict()

        source_fps = read_sequence_fps(sequence_dir, default_fps)
        previous_gray = None
        sequence_rows: list[dict] = []

        for zero_index, frame_path in enumerate(
            tqdm(frame_paths, desc=sequence_dir.name, leave=False)
        ):
            frame_number = zero_index + 1
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            features = frame_feature_vector(image, previous_gray)
            object_count = int(counts.get(frame_number, 0))

            sequence_rows.append(
                {
                    "sequence": sequence_dir.name,
                    "frame": frame_number,
                    "time_sec": zero_index / source_fps,
                    "source_fps": source_fps,
                    "object_count": object_count,
                    "event_present": int(
                        object_count >= int(experiment_cfg["event_min_objects"])
                    ),
                    **features,
                }
            )
            previous_gray = image

        for index, row in enumerate(sequence_rows):
            next_count = (
                sequence_rows[index + 1]["object_count"]
                if index + 1 < len(sequence_rows)
                else row["object_count"]
            )
            row["object_count_change"] = abs(next_count - row["object_count"])
            row["oracle_fps"] = oracle_required_fps(
                row["object_count"], next_count, row["motion_score"]
            )
            rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("Feature extraction produced no rows.")

    output_path = Path("data/frame_features.csv")
    output.to_csv(output_path, index=False)

    summary = (
        output.groupby("sequence")
        .agg(
            frames=("frame", "count"),
            events=("event_present", "sum"),
            mean_objects=("object_count", "mean"),
        )
        .reset_index()
    )
    summary.to_csv("results/dataset_summary.csv", index=False)

    print(f"Saved {len(output):,} frame records to {output_path}")
    print(f"Used {output['sequence'].nunique()} independent sequences")


if __name__ == "__main__":
    main()
