# Edge AI Adaptive Camera Sampling for Energy-Efficient Defense UAV Surveillance

A simple and reproducible dataset-replay study for evaluating whether an Edge AI controller can reduce UAV camera sampling, processing, and transmission costs while preserving surveillance-event detection.

## Research design

The study uses the official **VisDrone2019-VID validation set**, which contains real UAV-recorded video sequences and manual object annotations.

The workflow:

1. Downloads the official VisDrone video validation set.
2. Extracts lightweight image features:
   - frame-difference motion;
   - image entropy;
   - edge density;
   - intensity mean and variation.
3. Creates an offline oracle sampling label of 1, 5, 10, or 20 FPS from annotated scene activity.
4. Splits data by complete video sequence.
5. Trains one Random Forest Edge AI sampling controller.
6. Compares it with:
   - fixed 1 FPS;
   - fixed 5 FPS;
   - fixed 10 FPS;
   - fixed 20 FPS;
   - motion-threshold sampling.
7. Reports event recall, detection delay, sampled frames, transmission volume, and estimated energy.
8. Produces bootstrap 95% confidence intervals and publication-ready figures.

## Important scientific limitation

Energy is estimated with a transparent per-frame analytical model. It is not measured on a physical UAV or edge device. The paper must report the results as **estimated energy**, not hardware-measured battery consumption.

No classified defense data are used. Defense relevance is represented by aerial monitoring of people and vehicle activity in public UAV footage.

## RunPod requirements

Recommended:

- PyTorch RunPod image;
- Python 3.10 or later;
- at least 20 GB free persistent storage;
- GPU optional because no detector is trained.

The official validation archive is approximately 1.49 GB. Extracted frames require additional space.

## Run the experiment

```bash
cd /workspace
git clone https://github.com/drsaikrishnathota1/edge-ai-adaptive-sampling-defense-uav.git
cd edge-ai-adaptive-sampling-defense-uav

bash scripts/runpod_setup.sh
python run_experiment.py
```

To reuse an already downloaded and extracted dataset:

```bash
python run_experiment.py --skip-download
```

## Main outputs

```text
results/
├── dataset_summary.csv
├── controller_metrics.csv
├── controller_classification_report.csv
├── controller_confusion_matrix.csv
├── policy_metrics_by_sequence.csv
├── policy_summary.csv
├── feature_importance.csv
└── sequence_split.csv

figures/
├── energy_vs_event_recall.png
├── energy_reduction.png
├── detection_delay.png
└── feature_importance.png

models/
└── adaptive_sampling_rf.joblib
```

## Reproducibility controls

- fixed random seed;
- sequence-level train/test separation;
- saved sequence split;
- fixed baseline policies;
- bootstrap confidence intervals;
- configuration stored in `config.yaml`;
- generated results stored as CSV files.

## Proposed manuscript claim

The experiment tests whether an Edge AI controller can choose a lower camera sampling rate during quiet scenes and increase sampling during activity, reducing estimated sensing and communication energy while retaining event-level surveillance recall.

## Dataset citation

P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, and H. Ling, “Detection and Tracking Meet Drones Challenge,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 11, pp. 7380–7399, 2022.
