#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [ ! -d "edge-ai-adaptive-sampling-defense-uav" ]; then
  git clone https://github.com/drsaikrishnathota1/edge-ai-adaptive-sampling-defense-uav.git
fi

cd edge-ai-adaptive-sampling-defense-uav

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data results figures models

echo
echo "Setup complete."
echo "Run:"
echo "python run_experiment.py"
