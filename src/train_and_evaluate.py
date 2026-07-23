#!/usr/bin/env python3
"""Train and evaluate event-risk-driven adaptive camera sampling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "entropy",
    "edge_density",
    "motion_score",
]

RATES = (1, 5, 10, 20)


@dataclass
class PolicyResult:
    policy: str
    sequence: str
    fold: int
    total_events: int
    detected_events: int
    sampled_frames: int
    total_frames: int
    event_recall: float
    mean_detection_delay_sec: float
    energy_mj: float
    transmitted_frames: int
    mean_selected_fps: float


def normalize_rate(rate: int) -> int:
    return min(RATES, key=lambda candidate: abs(candidate - int(rate)))


def thresholds_from_quantiles(values: np.ndarray, quantiles: list[float]) -> tuple[float, ...]:
    clean = np.asarray(values, dtype=float)
    if clean.size == 0:
        raise ValueError("Cannot calibrate thresholds from an empty array.")
    return tuple(float(value) for value in np.quantile(clean, quantiles))


def rate_from_score(score: float, thresholds: tuple[float, ...]) -> int:
    low, medium, high = thresholds
    if score <= low:
        return 1
    if score <= medium:
        return 5
    if score <= high:
        return 10
    return 20


def bootstrap_ci(values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
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
    detection_window_sec: float,
    fold: int,
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

    event_indices = np.flatnonzero(
        sequence_df["event_present"].to_numpy(dtype=int) == 1
    )
    detection_window_frames = max(1, round(detection_window_sec * source_fps))

    detected_events = 0
    delays: list[float] = []
    for event_index in event_indices:
        event_end = min(total_frames - 1, event_index + detection_window_frames)
        hits = [
            sampled_index
            for sampled_index in sampled_indices
            if event_index <= sampled_index <= event_end
        ]
        if hits:
            detected_events += 1
            delays.append((hits[0] - event_index) / source_fps)

    total_events = len(event_indices)
    event_recall = detected_events / total_events if total_events else float("nan")
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    sampled_frames = len(sampled_indices)
    preview_energy = total_frames * float(energy_cfg["preview_mj_per_source_frame"])
    selected_frame_energy = sampled_frames * (
        float(energy_cfg["camera_mj_per_selected_frame"])
        + float(energy_cfg["feature_processing_mj_per_selected_frame"])
        + float(energy_cfg["transmission_mj_per_selected_frame"])
    )
    controller_energy = sampled_frames * float(
        energy_cfg["controller_mj_per_decision"]
    )
    energy_mj = preview_energy + selected_frame_energy + controller_energy

    return PolicyResult(
        policy=policy_name,
        sequence=str(sequence_df["sequence"].iloc[0]),
        fold=fold,
        total_events=total_events,
        detected_events=detected_events,
        sampled_frames=sampled_frames,
        total_frames=total_frames,
        event_recall=event_recall,
        mean_detection_delay_sec=mean_delay,
        energy_mj=energy_mj,
        transmitted_frames=sampled_frames,
        mean_selected_fps=float(np.mean(selected_rates)),
    )


def build_model(cfg: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(cfg["random_forest_trees"]),
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
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
    quantiles = list(map(float, exp_cfg["rate_quantiles"]))
    detection_window_sec = float(exp_cfg["detection_window_sec"])

    data = pd.read_csv("data/frame_features.csv")
    if data["sequence"].nunique() < 3:
        raise RuntimeError("At least three independent sequences are required.")
    if int(data["event_present"].sum()) == 0:
        raise RuntimeError("No new-track events were found in the selected data.")

    groups = data["sequence"].to_numpy()
    logo = LeaveOneGroupOut()

    all_true: list[float] = []
    all_predicted: list[float] = []
    policy_rows: list[dict] = []
    fold_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(data, groups=groups), start=1
    ):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()
        test_sequence = str(test_df["sequence"].iloc[0])

        model = build_model(exp_cfg, seed + fold)
        model.fit(train_df[FEATURE_COLUMNS], train_df["future_event_risk"])

        test_predictions = np.maximum(
            0.0, model.predict(test_df[FEATURE_COLUMNS])
        )
        all_true.extend(test_df["future_event_risk"].tolist())
        all_predicted.extend(test_predictions.tolist())

        train_risk_predictions = np.maximum(
            0.0, model.predict(train_df[FEATURE_COLUMNS])
        )
        ai_thresholds = thresholds_from_quantiles(
            train_risk_predictions, quantiles
        )
        visual_thresholds = thresholds_from_quantiles(
            train_df["motion_score"].to_numpy(), quantiles
        )

        def ai_policy(row: pd.Series) -> int:
            frame = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()])
            predicted_risk = max(0.0, float(model.predict(frame)[0]))
            return rate_from_score(predicted_risk, ai_thresholds)

        def visual_policy(row: pd.Series) -> int:
            return rate_from_score(float(row["motion_score"]), visual_thresholds)

        policies: list[tuple[str, Callable[[pd.Series], int]]] = [
            ("Fixed-1-FPS", lambda row: 1),
            ("Fixed-5-FPS", lambda row: 5),
            ("Fixed-10-FPS", lambda row: 10),
            ("Fixed-20-FPS", lambda row: 20),
            ("Visual-Threshold", visual_policy),
            ("Edge-AI-Adaptive", ai_policy),
        ]

        for sequence_name, sequence_df in test_df.groupby("sequence"):
            for policy_name, policy in policies:
                result = simulate_policy(
                    sequence_df=sequence_df,
                    policy_name=policy_name,
                    choose_rate=policy,
                    energy_cfg=energy_cfg,
                    detection_window_sec=detection_window_sec,
                    fold=fold,
                )
                policy_rows.append(asdict(result))

        fold_rows.append(
            {
                "fold": fold,
                "test_sequence": test_sequence,
                "train_frames": len(train_df),
                "test_frames": len(test_df),
                "train_sequences": train_df["sequence"].nunique(),
                "ai_threshold_1_to_5": ai_thresholds[0],
                "ai_threshold_5_to_10": ai_thresholds[1],
                "ai_threshold_10_to_20": ai_thresholds[2],
                "visual_threshold_1_to_5": visual_thresholds[0],
                "visual_threshold_5_to_10": visual_thresholds[1],
                "visual_threshold_10_to_20": visual_thresholds[2],
            }
        )

    true_values = np.asarray(all_true, dtype=float)
    predicted_values = np.asarray(all_predicted, dtype=float)
    correlation = spearmanr(true_values, predicted_values).statistic

    controller_metrics = {
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(true_values, predicted_values))),
        "r2": float(r2_score(true_values, predicted_values)),
        "spearman_rho": float(correlation) if np.isfinite(correlation) else float("nan"),
        "sequences": int(data["sequence"].nunique()),
        "frames": int(len(data)),
        "new_track_events": int(data["event_present"].sum()),
        "positive_risk_frames": int((data["future_event_risk"] > 0).sum()),
    }
    pd.DataFrame([controller_metrics]).to_csv(
        "results/controller_metrics.csv", index=False
    )

    policy_metrics = pd.DataFrame(policy_rows)
    policy_metrics.to_csv("results/policy_metrics_by_sequence.csv", index=False)
    pd.DataFrame(fold_rows).to_csv("results/cross_validation_folds.csv", index=False)

    bootstrap_iterations = int(exp_cfg["bootstrap_iterations"])
    summary_rows: list[dict] = []
    for policy_name, policy_df in policy_metrics.groupby("policy"):
        recall_ci = bootstrap_ci(
            policy_df["event_recall"].tolist(), bootstrap_iterations, seed
        )
        delay_ci = bootstrap_ci(
            policy_df["mean_detection_delay_sec"].tolist(),
            bootstrap_iterations,
            seed + 1,
        )
        energy_ci = bootstrap_ci(
            policy_df["energy_mj"].tolist(), bootstrap_iterations, seed + 2
        )

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
                "total_events": int(policy_df["total_events"].sum()),
                "detected_events": int(policy_df["detected_events"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    fixed_20_energy = float(
        summary.loc[
            summary["policy"] == "Fixed-20-FPS", "energy_mj_mean"
        ].iloc[0]
    )
    summary["energy_reduction_vs_20fps_pct"] = (
        100.0 * (fixed_20_energy - summary["energy_mj_mean"]) / fixed_20_energy
    )
    summary.to_csv("results/policy_summary.csv", index=False)

    final_model = build_model(exp_cfg, seed)
    final_model.fit(data[FEATURE_COLUMNS], data["future_event_risk"])
    final_predictions = np.maximum(
        0.0, final_model.predict(data[FEATURE_COLUMNS])
    )
    final_thresholds = thresholds_from_quantiles(final_predictions, quantiles)
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURE_COLUMNS,
            "risk_thresholds": final_thresholds,
            "architecture": "low-resolution preview controls selected full-resolution frames",
        },
        "models/adaptive_sampling_rf.joblib",
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv("results/feature_importance.csv", index=False)

    for obsolete in (
        "results/controller_classification_report.csv",
        "results/controller_confusion_matrix.csv",
        "results/sequence_split.csv",
    ):
        Path(obsolete).unlink(missing_ok=True)

    print("Controller risk-prediction metrics:")
    print(pd.DataFrame([controller_metrics]).round(4).to_string(index=False))
    print("\nPolicy summary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
