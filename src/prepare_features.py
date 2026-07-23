#!/usr/bin/env python3
"""Prepare low-resolution preview features and annotation-derived event risk."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

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

    frame = pd.read_csv(path, header=None)
    if frame.shape[1] < 10:
        raise ValueError(f"Unexpected annotation format in {path}")

    frame = frame.iloc[:, :10]
    frame.columns = VISDRONE_COLUMNS
    valid = (frame["score"] != 0) & frame["category"].isin(target_categories)
    return frame.loc[valid].copy()


def frame_feature_vector(
    gray: np.ndarray,
    previous_gray: np.ndarray | None,
) -> dict[str, float]:
    """Extract features from a low-resolution always-on preview stream."""
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


def add_future_risk(sequence_rows: list[dict], horizon_frames: int) -> None:
    new_tracks = np.asarray(
        [row["new_track_count"] for row in sequence_rows], dtype=float
    )
    count_changes = np.asarray(
        [row["object_count_change"] for row in sequence_rows], dtype=float
    )

    for index, row in enumerate(sequence_rows):
        start = index + 1
        end = min(len(sequence_rows), start + horizon_frames)
        if start >= end:
            row["future_event_risk"] = 0.0
            continue

        # New target entries dominate; count changes add secondary scene activity.
        row["future_event_risk"] = float(
            new_tracks[start:end].sum() + 0.25 * count_changes[start:end].sum()
        )


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

    sequence_dirs = sorted([path for path in sequences_dir.iterdir() if path.is_dir()])
    max_sequences = int(dataset_cfg["max_sequences"])
    if max_sequences > 0:
        sequence_dirs = sequence_dirs[:max_sequences]

    if not sequence_dirs:
        raise RuntimeError("No sequence directories were found.")

    target_categories = set(map(int, experiment_cfg["target_categories"]))
    max_frames = int(dataset_cfg["max_frames_per_sequence"])
    default_fps = int(dataset_cfg["default_source_fps"])
    event_horizon_sec = float(experiment_cfg["event_horizon_sec"])

    rows: list[dict] = []

    for sequence_dir in sequence_dirs:
        frame_paths = sorted(sequence_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(sequence_dir.glob("*.png"))
        if max_frames > 0:
            frame_paths = frame_paths[:max_frames]

        annotation_path = annotations_dir / f"{sequence_dir.name}.txt"
        annotations = read_annotations(annotation_path, target_categories)
        grouped_ids = (
            annotations.groupby("frame")["target_id"]
            .apply(lambda values: set(map(int, values)))
            .to_dict()
        )

        source_fps = read_sequence_fps(sequence_dir, default_fps)
        horizon_frames = max(1, round(event_horizon_sec * source_fps))
        previous_gray = None
        previous_count = 0
        seen_ids: set[int] = set()
        sequence_rows: list[dict] = []

        for zero_index, frame_path in enumerate(
            tqdm(frame_paths, desc=sequence_dir.name, leave=False)
        ):
            frame_number = zero_index + 1
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            current_ids = grouped_ids.get(frame_number, set())
            if zero_index == 0:
                new_ids: set[int] = set()
                seen_ids.update(current_ids)
            else:
                new_ids = current_ids - seen_ids
                seen_ids.update(current_ids)

            object_count = len(current_ids)
            object_count_change = abs(object_count - previous_count)
            features = frame_feature_vector(image, previous_gray)

            sequence_rows.append(
                {
                    "sequence": sequence_dir.name,
                    "frame": frame_number,
                    "time_sec": zero_index / source_fps,
                    "source_fps": source_fps,
                    "object_count": object_count,
                    "object_count_change": object_count_change,
                    "new_track_count": len(new_ids),
                    "event_present": int(len(new_ids) > 0),
                    **features,
                }
            )

            previous_gray = image
            previous_count = object_count

        add_future_risk(sequence_rows, horizon_frames)
        rows.extend(sequence_rows)

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("Feature extraction produced no rows.")

    output_path = Path("data/frame_features.csv")
    output.to_csv(output_path, index=False)

    summary = (
        output.groupby("sequence")
        .agg(
            frames=("frame", "count"),
            new_track_events=("event_present", "sum"),
            new_tracks=("new_track_count", "sum"),
            mean_objects=("object_count", "mean"),
            positive_risk_frames=("future_event_risk", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    summary.to_csv("results/dataset_summary.csv", index=False)

    print(f"Saved {len(output):,} preview-frame records to {output_path}")
    print(f"Used {output['sequence'].nunique()} independent sequences")
    print(f"Recorded {int(output['event_present'].sum()):,} new-track events")


if __name__ == "__main__":
    main()
