#!/usr/bin/env python3
"""Create publication-ready figures from saved CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def save_figure(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        yaml.safe_load(handle)

    Path("figures").mkdir(exist_ok=True)
    summary = pd.read_csv("results/policy_summary.csv")
    importance = pd.read_csv("results/feature_importance.csv")

    ordered = [
        "Fixed-1-FPS",
        "Fixed-5-FPS",
        "Fixed-10-FPS",
        "Fixed-20-FPS",
        "Motion-Threshold",
        "Edge-AI-Adaptive",
    ]
    summary["policy"] = pd.Categorical(
        summary["policy"], categories=ordered, ordered=True
    )
    summary = summary.sort_values("policy")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        summary["energy_mj_mean"],
        summary["event_recall_mean"],
        s=90,
    )
    for _, row in summary.iterrows():
        plt.annotate(
            str(row["policy"]),
            (row["energy_mj_mean"], row["event_recall_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Estimated energy per sequence (mJ)")
    plt.ylabel("Event recall")
    plt.title("Energy–surveillance performance trade-off")
    plt.grid(alpha=0.25)
    save_figure("figures/energy_vs_event_recall.png")

    plt.figure(figsize=(9, 5))
    plt.bar(
        summary["policy"].astype(str),
        summary["energy_reduction_vs_20fps_pct"],
    )
    plt.axhline(0, linewidth=1)
    plt.ylabel("Energy reduction versus fixed 20 FPS (%)")
    plt.xlabel("Sampling policy")
    plt.title("Estimated energy reduction")
    plt.xticks(rotation=25, ha="right")
    save_figure("figures/energy_reduction.png")

    plt.figure(figsize=(9, 5))
    plt.bar(
        summary["policy"].astype(str),
        summary["detection_delay_sec_mean"],
    )
    plt.ylabel("Mean event detection delay (s)")
    plt.xlabel("Sampling policy")
    plt.title("Event detection delay")
    plt.xticks(rotation=25, ha="right")
    save_figure("figures/detection_delay.png")

    plt.figure(figsize=(8, 5))
    plt.barh(
        importance["feature"],
        importance["importance"],
    )
    plt.xlabel("Random Forest importance")
    plt.ylabel("Feature")
    plt.title("Edge AI controller feature importance")
    plt.gca().invert_yaxis()
    save_figure("figures/feature_importance.png")

    print("Saved four figures in figures/")


if __name__ == "__main__":
    main()
