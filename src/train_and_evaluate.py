#!/usr/bin/env python3
"""Train the Edge AI controller and evaluate all sampling policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "entropy",
    "edge_density",
]

RATES = (1, 5, 10, 20)


@dataclass
class PolicyResult:
    policy: str
    sequence: str
    sampled_frames: int
    total_frames: int
    event_recall: float
    mean_detection_delay_sec: float
    energy_mj: float
    transmitted_frames: int
    mean_selected_fps: float


def normalize_rate(rate: int) -> int:
    return min(RATES, key=lambda candidate: abs(candidate - int(rate)))


def contiguous_event_episodes(event_values: np.ndarray) -> list[tuple[int, int]]:
    episodes: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(event_values):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            episodes.append((start, index - 1))
            start = None
    if start is not None:
        episodes.append((start, len(event_values) - 1))
    return episodes


def bootstrap_ci(values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = [
        float(np.mean(rng.choice(clean, size=len(clean), replace=True)))
        for _ in range(iterations)
    ]
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def simulate_policy(
    sequence_df: pd.DataFrame,
    policy_name: str,
    choose_rate: Callable[[pd.Series], int],
    energy_cfg: dict,
) -> PolicyResult:
    sequence_df = sequence_df.sort_values("frame").reset_index(drop=True)
    source_fps = int(sequence_df["source_fps"].iloc[0])
    total_frames = len(sequence_df)

    sampled_indices: list[int] = []
    selected_rates: list[int] = []
    cursor = 0.0

    while cursor < total_frames:
        frame_index = int(round(cursor))

        if sampled_indices and frame_index <= sampled_indices[-1]:
            frame_index = sampled_indices[-1] + 1

        if frame_index >= total_frames:
            break

        row = sequence_df.iloc[frame_index]
        selected_rate = normalize_rate(choose_rate(row))
        sampled_indices.append(frame_index)
        selected_rates.append(selected_rate)

        cursor += source_fps / selected_rate
    episodes = contiguous_event_episodes(
        sequence_df["event_present"].to_numpy(dtype=int)
    )

    detected = 0
    delays: list[float] = []
    for start, end in episodes:
        hits = [index for index in sampled_indices if start <= index <= end]
        if hits:
            detected += 1
            delays.append((hits[0] - start) / source_fps)

    event_recall = detected / len(episodes) if episodes else 1.0
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    sampled_frames = len(sampled_indices)
    per_frame_mj = (
        float(energy_cfg["camera_mj_per_frame"])
        + float(energy_cfg["feature_processing_mj_per_frame"])
        + float(energy_cfg["transmission_mj_per_frame"])
    )
    energy_mj = (
        sampled_frames * per_frame_mj
        + sampled_frames * float(energy_cfg["controller_mj_per_decision"])
    )

    return PolicyResult(
        policy=policy_name,
        sequence=str(sequence_df["sequence"].iloc[0]),
        sampled_frames=sampled_frames,
        total_frames=total_frames,
        event_recall=event_recall,
        mean_detection_delay_sec=mean_delay,
        energy_mj=energy_mj,
        transmitted_frames=sampled_frames,
        mean_selected_fps=float(np.mean(selected_rates)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = int(cfg["seed"])
    exp_cfg = cfg["experiment"]
    energy_cfg = cfg["energy_model"]
    threshold_cfg = cfg["visual_threshold_policy"]

    data = pd.read_csv("data/frame_features.csv")
    groups = data["sequence"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(exp_cfg["test_size"]),
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(data, groups=groups))
    train_df = data.iloc[train_idx].copy()
    test_df = data.iloc[test_idx].copy()

    model = RandomForestClassifier(
        n_estimators=int(exp_cfg["random_forest_trees"]),
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["oracle_fps"])

    predictions = model.predict(test_df[FEATURE_COLUMNS])
    controller_metrics = {
        "accuracy": float(accuracy_score(test_df["oracle_fps"], predictions)),
        "macro_f1": float(
            f1_score(test_df["oracle_fps"], predictions, average="macro")
        ),
        "train_sequences": int(train_df["sequence"].nunique()),
        "test_sequences": int(test_df["sequence"].nunique()),
        "train_frames": int(len(train_df)),
        "test_frames": int(len(test_df)),
    }
    pd.DataFrame([controller_metrics]).to_csv(
        "results/controller_metrics.csv", index=False
    )

    report = classification_report(
        test_df["oracle_fps"],
        predictions,
        labels=list(RATES),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        "results/controller_classification_report.csv"
    )

    confusion = confusion_matrix(
        test_df["oracle_fps"], predictions, labels=list(RATES)
    )
    pd.DataFrame(
        confusion,
        index=[f"true_{rate}" for rate in RATES],
        columns=[f"pred_{rate}" for rate in RATES],
    ).to_csv("results/controller_confusion_matrix.csv")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS},
        "models/adaptive_sampling_rf.joblib",
    )

    low_threshold = float(threshold_cfg["low_threshold"])
    high_threshold = float(threshold_cfg["high_threshold"])

    def ai_policy(row: pd.Series) -> int:
        frame = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()])
        return int(model.predict(frame)[0])

    def threshold_policy(row: pd.Series) -> int:
        activity_score = (
            100.0 * float(row["edge_density"])
            + 2.0 * float(row["entropy"])
        )

        if activity_score < low_threshold:
            return 1
        if activity_score < high_threshold:
            return 5
        return 20

    policies: list[tuple[str, Callable[[pd.Series], int]]] = [
        ("Fixed-1-FPS", lambda row: 1),
        ("Fixed-5-FPS", lambda row: 5),
        ("Fixed-10-FPS", lambda row: 10),
        ("Fixed-20-FPS", lambda row: 20),
        ("Visual-Threshold", threshold_policy),
        ("Edge-AI-Adaptive", ai_policy),
    ]

    sequence_results: list[dict] = []
    for sequence_name, sequence_df in test_df.groupby("sequence"):
        for name, policy in policies:
            result = simulate_policy(
                sequence_df, name, policy, energy_cfg
            )
            sequence_results.append(result.__dict__)

    sequence_metrics = pd.DataFrame(sequence_results)
    sequence_metrics.to_csv("results/policy_metrics_by_sequence.csv", index=False)

    bootstrap_iterations = int(exp_cfg["bootstrap_iterations"])
    summary_rows: list[dict] = []
    for policy_name, policy_df in sequence_metrics.groupby("policy"):
        recall_ci = bootstrap_ci(
            policy_df["event_recall"].tolist(),
            bootstrap_iterations,
            seed,
        )
        energy_ci = bootstrap_ci(
            policy_df["energy_mj"].tolist(),
            bootstrap_iterations,
            seed + 1,
        )
        delay_values = policy_df["mean_detection_delay_sec"].tolist()
        delay_ci = bootstrap_ci(delay_values, bootstrap_iterations, seed + 2)

        summary_rows.append(
            {
                "policy": policy_name,
                "event_recall_mean": policy_df["event_recall"].mean(),
                "event_recall_ci_low": recall_ci[0],
                "event_recall_ci_high": recall_ci[1],
                "detection_delay_sec_mean": policy_df[
                    "mean_detection_delay_sec"
                ].mean(),
                "detection_delay_ci_low": delay_ci[0],
                "detection_delay_ci_high": delay_ci[1],
                "energy_mj_mean": policy_df["energy_mj"].mean(),
                "energy_ci_low": energy_ci[0],
                "energy_ci_high": energy_ci[1],
                "sampled_frames_mean": policy_df["sampled_frames"].mean(),
                "mean_selected_fps": policy_df["mean_selected_fps"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    fixed_20_energy = float(
        summary.loc[summary["policy"] == "Fixed-20-FPS", "energy_mj_mean"].iloc[0]
    )
    summary["energy_reduction_vs_20fps_pct"] = (
        100.0 * (fixed_20_energy - summary["energy_mj_mean"]) / fixed_20_energy
    )
    summary.to_csv("results/policy_summary.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv("results/feature_importance.csv", index=False)

    split = pd.DataFrame(
        {
            "sequence": sorted(data["sequence"].unique()),
            "split": [
                "test" if name in set(test_df["sequence"]) else "train"
                for name in sorted(data["sequence"].unique())
            ],
        }
    )
    split.to_csv("results/sequence_split.csv", index=False)

    print("Controller metrics:")
    print(pd.DataFrame([controller_metrics]).to_string(index=False))
    print("\nPolicy summary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
