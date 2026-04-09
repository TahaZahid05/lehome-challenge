# LeHome Challenge: SmolVLA Training & Evaluation (Local)

This document summarizes the transition from Diffusion Policy to the **SmolVLA** (Vision-Language-Action) architecture and our ongoing efforts to optimize garment folding performance.

## 1. Current Status: Robustness Training (In Progress)
We are currently executing a "Robustness Run" (April 2026) using an optimized training configuration designed to address the generalization gap on "Unseen" garments.

- **Config Path:** `configs/train_smolvla_aug.yaml`
- **Batch Size:** 64 (Increased from 16 to maximize RTX 4090 utilization ~18.8GB VRAM)
- **Data Augmentation (Visual IQ):**
    - **Color Jitter:** Randomly varying brightness, contrast, and saturation ([0.0, 1.5]). The wide saturation range forces the model to ignore color bias (simulating grayscale-to-vibrant logic).
    - **Random Affine:** Random ±5° rotations and ±5% translations to simulate camera bumps and varying garment placements.
- **Hardware Optimization:** `num_workers: 8` for high-throughput data loading.

## 2. Architecture & Baseline Strategy
- **Policy Type:** `smolvla` (LeRobot based)
- **Backbone:** `SmolVLM2-500M-Video-Instruct`
- **Training Strategy:** **Partial Fine-Tuning (Expert-Only)**
    - Vision Encoder: Frozen
    - VLM Backbone: Frozen
    - Action Expert: Trained (99.8M learnable parameters)
- **Dataset:** `four_types_merged_with_depth` (266K frames, 1,000 episodes)

## 3. Local Evaluation Results (Initial Baseline)
Results from the initial **60k-step / Batch 16** (No Augmentation) run:

| Garment Type | SmolVLA (Local) | Baseline (Estimate) |
| :--- | :--- | :--- |
| **Short Pants** | **73.33%** | 68% |
| **Long-sleeved Tops** | **56.67%** | 62% |
| **Long Pants** | **35.00%** | 60% |
| **Short-sleeved Tops** | **11.67%** | 38% |

*Note: These are local evaluation metrics. Official results will depend on the organizers' hold-out test set.*

## 4. Future Work & Research Directions
We are currently brainstorming modular ways to improve 3D visual perception and cross-view consistency without breaking the pre-trained VLM features.

### A. Advanced 3D Integration
- **Depth Heatmaps:** Investigating the injection of depth as a "Colorized Heatmap" (e.g., Magma/Jet colormaps). This allows the frozen VLM to perceive 3D geometry using its existing 3-channel RGB visual priors.

### B. Cross-View Consistency
- **Geometric Conditioning:** Providing **Camera Extrinsics** (position and orientation) as explicit input features. This is critical for the "Eye-in-Hand" (moving wrist cameras) to help the model anchor its visual tokens in a unified 3D workspace.
- **Consistency Loss:** Exploring auxiliary loss terms to ensure the model's perception of the garment remains spatially consistent across the Top, Left, and Right camera feeds.

---
*Last Updated: 2026-04-09*