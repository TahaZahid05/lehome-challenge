# LeHome Challenge: SmolVLA Training & Evaluation (Local)

This document summarizes the transition from Diffusion Policy to the **SmolVLA** (Vision-Language-Action) architecture and the initial results obtained from local evaluation.

## 1. Architecture & Training Setup
The SmolVLA model was chosen to address the challenge's requirement for semantic reasoning on RGB images without explicit garment labels.

- **Policy Type:** `smolvla` (LeRobot based)
- **Backbone:** `SmolVLM2-500M-Video-Instruct`
- **Training Strategy:** **Partial Fine-Tuning (Expert-Only)**
  - Vision Encoder: Frozen
  - VLM Backbone: Frozen
  - Action Expert: Trained (99.8M learnable parameters)
- **Dataset:** `four_types_merged_with_depth` (266K frames, 1,000 episodes)
- **Training Duration:** 60,000 steps (Batch Size 16)
- **Hardware Profile:** RTX 4090

## 2. Local Evaluation Results
The following results were obtained from evaluating the **60k-step checkpoint** against the release test set in the local simulation environment.

| Garment Type | SmolVLA (Local) | Baseline (Estimate) |
| :--- | :--- | :--- |
| **Short Pants** | **73.33%** | 68% |
| **Long-sleeved Tops** | **56.67%** | 62% |
| **Long Pants** | **35.00%** | 60% |
| **Short-sleeved Tops** | **11.67%** | 38% |

*Note: These are local evaluation metrics. Official results will depend on the organizers' hold-out test set and evaluation environment.*

## 3. Key Technical Fixes
- **Video Naming Improvement:** Modified `scripts/utils/eval_utils.py` and `scripts/utils/evaluation.py` to include the `garment_name` in filenames. This ensures all evaluation footage is uniquely labeled (e.g., `Shirt_0_episode1_...`) and prevents overwriting during batch runs.
- **Log Collection:** Implemented shell redirection for evaluation commands to ensure all success/failure metrics are persistently saved for analysis.

---
*Last Updated: 2026-03-28*