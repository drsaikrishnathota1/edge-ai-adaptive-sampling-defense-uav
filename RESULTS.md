# Final Experimental Results

## Experiment

The evaluation used seven independent VisDrone UAV video sequences containing:

- 2,846 preview frames
- 266 new-track surveillance events
- Leave-one-sequence-out cross-validation
- Fixed-rate, visual-threshold, and Edge AI adaptive sampling policies

## Main Result

The Edge AI adaptive policy achieved:

- Event recall: 99.65%
- Detected events: 265 of 266
- Mean detection delay: 0.0418 seconds
- Mean selected sampling rate: 11.06 FPS
- Estimated energy reduction versus fixed 20 FPS: 54.76%

## Baseline Comparison

| Policy | Event Recall | Detection Delay | Energy Reduction vs. 20 FPS |
|---|---:|---:|---:|
| Fixed 1 FPS | 50.45% | 0.2331 s | 92.55% |
| Fixed 5 FPS | 99.05% | 0.0845 s | 73.10% |
| Fixed 10 FPS | 99.65% | 0.0328 s | 48.68% |
| Fixed 20 FPS | 99.65% | 0.0084 s | 0.00% |
| Visual Threshold | 83.95% | 0.1088 s | 70.10% |
| Edge AI Adaptive | 99.65% | 0.0418 s | 54.76% |

## Interpretation

The adaptive policy preserved the event recall of fixed 10 FPS and fixed
20 FPS while reducing estimated energy by 54.76% relative to fixed 20 FPS.

Compared with fixed 10 FPS, the adaptive policy used approximately 11.9%
less estimated energy, with a small increase in mean detection delay.

## Limitations

- Energy consumption was analytically estimated rather than measured on physical hardware.
- The study used public UAV footage rather than operational defense data.
- Direct future-risk regression performance was limited, with R² = -0.4801
  and Spearman correlation = 0.1194.
- Results represent dataset-replay validation and not an actual UAV flight test.

## Reproducibility

Git commit:

`93932ffaef6452ccb2e97ab08345a28adece9a00`
