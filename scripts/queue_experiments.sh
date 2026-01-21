#!/bin/bash

# Initial Models for comparison (Scheduler vs Muon vs Activation)
# Run sequentially to avoid OOM, or comment out to run specific ones.

echo "Starting Experiment 3: Spectral + Plateau..."
python scripts/train.py --config configs/experiments/exp3_spectral_plateau.yaml > exp3.log 2>&1
echo "Experiment 3 Finished."

echo "Starting Experiment 4: ResNet18 + Plateau..."
python scripts/train.py --config configs/experiments/exp4_resnet_plateau.yaml --model-config configs/model_resnet18.yaml > exp4.log 2>&1
echo "Experiment 4 Finished."

echo "Starting Experiment 5: Spectral + Muon..."
python scripts/train.py --config configs/experiments/exp5_spectral_muon.yaml > exp5.log 2>&1
echo "Experiment 5 Finished."

echo "Starting Experiment 6: ResNet18 + Muon..."
python scripts/train.py --config configs/experiments/exp6_resnet_muon.yaml --model-config configs/model_resnet18.yaml > exp6.log 2>&1
echo "Experiment 6 Finished."

echo "Starting Experiment 7: Spectral + SiLU..."
python scripts/train.py --config configs/experiments/exp7_spectral_silu.yaml > exp7.log 2>&1
echo "Experiment 7 Finished."

echo "Starting Experiment 8: Spectral + Tanh..."
python scripts/train.py --config configs/experiments/exp8_spectral_tanh.yaml > exp8.log 2>&1
echo "Experiment 8 Finished."

echo "All Experiments Completed!"
