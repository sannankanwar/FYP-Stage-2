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


echo "Starting Experiment 9: PINN + SiLU..."
python scripts/train.py --config configs/experiments/exp9_pinn_silu.yaml > exp9.log 2>&1
echo "Experiment 9 Finished."

echo "Starting Experiment 10: PINN + Tanh..."
python scripts/train.py --config configs/experiments/exp10_pinn_tanh.yaml > exp10.log 2>&1
echo "Experiment 10 Finished."

echo "Starting Experiment 11: Spectral + Stdz + MSE..."
python scripts/train.py --config configs/experiments/exp11_spectral_std_mse.yaml > exp11.log 2>&1
echo "Experiment 11 Finished."

echo "Starting Experiment 12: Spectral + Stdz + Physics..."
python scripts/train.py --config configs/experiments/exp12_spectral_std_physics.yaml > exp12.log 2>&1
echo "Experiment 12 Finished."

echo "Starting Experiment 13: ResNet + Stdz + MSE..."
python scripts/train.py --config configs/experiments/exp13_resnet_std_mse.yaml --model-config configs/model_resnet18.yaml > exp13.log 2>&1
echo "Experiment 13 Finished."

echo "Starting Experiment 14: ResNet + Stdz + Physics..."
python scripts/train.py --config configs/experiments/exp14_resnet_std_physics.yaml --model-config configs/model_resnet18.yaml > exp14.log 2>&1
echo "Experiment 14 Finished."

echo "All Experiments Completed!"
