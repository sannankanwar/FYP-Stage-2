#!/usr/bin/env python3
"""
Standalone Demo Script for Noise Pipeline.

This script demonstrates the noise pipeline module with various configurations.
Run this script to visualize different noise effects on metalens phase maps.

Usage:
    python scripts/demo_noise_pipeline.py [--output-dir PATH] [--seed SEED]

Examples:
    # Run with default settings
    python scripts/demo_noise_pipeline.py
    
    # Custom output directory and seed
    python scripts/demo_noise_pipeline.py --output-dir ./my_outputs --seed 123
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from src.noise import (
    NoisePipeline,
    get_default_noise_config,
    generate_wrapped_phase_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Noise Pipeline Demo')
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./outputs/noise_demo',
        help='Directory to save output figures'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--resolution', 
        type=int, 
        default=1024,
        help='Image resolution (H=W)'
    )
    parser.add_argument(
        '--wavelength-nm',
        type=float,
        default=532.0,
        help='Wavelength in nanometers (default: 532 nm green)'
    )
    parser.add_argument(
        '--xc',
        type=float,
        default=0.0,
        help='Center X in micrometers'
    )
    parser.add_argument(
        '--yc',
        type=float,
        default=0.0,
        help='Center Y in micrometers'
    )
    parser.add_argument(
        '--fov',
        type=float,
        default=200.0,
        help='Field of View (S) in micrometers'
    )
    parser.add_argument(
        '--focal-length',
        type=float,
        default=500.0,
        help='Focal Length in micrometers'
    )
    return parser.parse_args()


def demo_individual_components(
    phi_clean: torch.Tensor,
    img2_clean: torch.Tensor,
    config: dict,
    output_dir: Path,
):
    """Demo each noise component individually from clean baseline."""
    
    pipeline = NoisePipeline(config)
    
    # Get all components in their canonical order
    component_names = pipeline.pipeline_order
    
    results = {}
    
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = 'twilight'
    
    print("\n--- Individual Component Effects ---")
    
    for name in component_names:
        # Only demo components that are enabled or skip 'wrap_phase' if desired, 
        # but let's show everything the pipeline intends to do.
        if name == 'wrap_phase': continue
            
        print(f"  Processing {name}...")
        
        phi_after, img2_after = pipeline.apply_single_component(
            name, phi_clean.clone(), img2_clean.clone()
        )
        results[name] = phi_after
        
        # Compute stats
        diff = (phi_after - phi_clean).abs()
        print(f"    Mean |Δφ|: {diff.mean().item():.4f} rad")
        print(f"    Max |Δφ|:  {diff.max().item():.4f} rad")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        title = name.replace('_', ' ').title()
        fig.suptitle(f'Noise Component: {title}', fontsize=14, fontweight='bold')
        
        # Before
        im0 = axes[0].imshow(phi_clean[0].cpu().numpy(), cmap=cmap, norm=phase_norm)
        axes[0].set_title('Before (Clean Baseline)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='φ (rad)', shrink=0.8)
        
        # After
        im1 = axes[1].imshow(phi_after[0].cpu().numpy(), cmap=cmap, norm=phase_norm)
        axes[1].set_title(f'After {title}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='φ (rad)', shrink=0.8)
        
        # Difference
        diff_np = diff[0].cpu().numpy()
        im2 = axes[2].imshow(diff_np, cmap='hot')
        axes[2].set_title('|Δφ| (Difference)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='|Δφ| (rad)', shrink=0.8)
        
        plt.tight_layout()
        filepath = output_dir / f'component_{name}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {filepath}")
    
    return results


def demo_composite_pipeline(
    phi_clean: torch.Tensor,
    img2_clean: torch.Tensor,
    config: dict,
    output_dir: Path,
):
    """Demo the full composite noise pipeline."""
    
    print("\n--- Composite Pipeline ---")
    
    pipeline = NoisePipeline(config)
    phi_out, img2_out, step_outputs = pipeline.apply(phi_clean.clone(), img2_clean.clone())
    
    print(f"  Steps applied: {list(step_outputs.keys())}")
    
    diff = (phi_out - phi_clean).abs()
    print(f"  Mean |Δφ|: {diff.mean().item():.4f} rad")
    print(f"  Max |Δφ|:  {diff.max().item():.4f} rad")
    
    # Visualization
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = 'twilight'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Composite Noise Pipeline', fontsize=14, fontweight='bold')
    
    im0 = axes[0].imshow(phi_clean[0].cpu().numpy(), cmap=cmap, norm=phase_norm)
    axes[0].set_title('Before (Clean Baseline)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='φ (rad)', shrink=0.8)
    
    im1 = axes[1].imshow(phi_out[0].cpu().numpy(), cmap=cmap, norm=phase_norm)
    axes[1].set_title('After All Components')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='φ (rad)', shrink=0.8)
    
    diff_np = diff[0].cpu().numpy()
    im2 = axes[2].imshow(diff_np, cmap='hot')
    axes[2].set_title('|Δφ| (Difference)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='|Δφ| (rad)', shrink=0.8)
    
    plt.tight_layout()
    filepath = output_dir / 'composite_pipeline.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")
    
    # Summary figure
    create_summary_figure(phi_clean, step_outputs, phi_out, output_dir)
    
    return phi_out, step_outputs


def create_summary_figure(phi_clean, step_outputs, phi_composite, output_dir):
    """Create a summary figure showing pipeline progression."""
    
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = 'twilight'
    
    # Plots needed: 1 (clean) + len(step_outputs)
    n_plots = len(step_outputs) + 1
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    fig.suptitle('Noise Pipeline Progression', fontsize=16, fontweight='bold')
    
    # 0. Clean baseline
    axes[0].imshow(phi_clean[0].cpu().numpy(), cmap=cmap, norm=phase_norm)
    axes[0].set_title('0. Clean Baseline')
    axes[0].axis('off')
    
    # 1..N. Each step applied
    for i, (name, data) in enumerate(step_outputs.items()):
        ax = axes[i + 1]
        ax.imshow(data['after_phi'][0].cpu().numpy(), cmap=cmap, norm=phase_norm)
        ax.set_title(f'{i+1}. After {name.replace("_", " ").title()}')
        ax.axis('off')
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    filepath = output_dir / 'pipeline_progression.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def demo_parameter_sweep(
    phi_clean: torch.Tensor,
    img2_clean: torch.Tensor,
    base_config: dict,
    output_dir: Path,
):
    """Demo different noise strengths for key components."""
    
    print("\n--- Parameter Sweep Demo ---")
    
    # Sweep fabrication noise amplitude
    amplitudes = [0.1, 0.3, 0.5, 0.8, 1.2]
    
    fig, axes = plt.subplots(1, len(amplitudes), figsize=(3 * len(amplitudes), 3))
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = 'twilight'
    
    for i, amp in enumerate(amplitudes):
        config = get_default_noise_config()
        config['seed'] = base_config['seed']
        config['fabrication_grf']['enabled'] = True
        config['fabrication_grf']['amplitude_rad'] = amp
        config['wrap_phase']['enabled'] = True
        
        pipeline = NoisePipeline(config)
        phi_out, _ = pipeline.apply_single_component('fabrication_grf', phi_clean.clone())
        # Apply wrap phase
        phi_out, _ = pipeline.apply_single_component('wrap_phase', phi_out)
        
        axes[i].imshow(phi_out[0].cpu().numpy(), cmap=cmap, norm=phase_norm)
        axes[i].set_title(f'A = {amp} rad')
        axes[i].axis('off')
    
    fig.suptitle('Fabrication GRF: Amplitude Sweep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = output_dir / 'sweep_fabrication_amplitude.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NOISE PIPELINE DEMO SCRIPT")
    print("=" * 70)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    
    # Parameters
    H = W = args.resolution
    wavelength_um = args.wavelength_nm / 1000.0  # Convert to micrometers
    
    metalens_params = {
        'xc': args.xc,            # Center X (µm)
        'yc': args.yc,            # Center Y (µm)
        'S': args.fov,            # Field of view (µm)
        'focal_length': args.focal_length,   # Focal length (µm)
        'wavelength': wavelength_um,
    }
    
    print(f"\nMetalens Parameters:")
    print(f"  Resolution: {H}x{W}")
    for k, v in metalens_params.items():
        print(f"  {k}: {v}")
    
    # Generate baseline
    print("\nGenerating baseline phase map...")
    phi_clean, img2_clean = generate_wrapped_phase_image(
        **metalens_params, H=H, W=W, device=device
    )
    print(f"  φ range: [{phi_clean.min().item():.3f}, {phi_clean.max().item():.3f}]")
    
    # Configure noise
    config = get_default_noise_config()
    config['seed'] = args.seed
    
    # Enable all components using defaults defined in noise_pipeline.py
    config['coordinate_warp']['enabled'] = True
    config['fabrication_grf']['enabled'] = True
    config['structured_sinusoid']['enabled'] = True
    config['sensor_grain']['enabled'] = True
    config['dead_pixels']['enabled'] = True
    
    config['wrap_phase']['enabled'] = True
    
    # Run demos
    demo_individual_components(phi_clean, img2_clean, config, output_dir)
    demo_composite_pipeline(phi_clean, img2_clean, config, output_dir)
    demo_parameter_sweep(phi_clean, img2_clean, config, output_dir)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
