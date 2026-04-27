# LeHome Challenge: Bi-Manual Garment Folding Summary

This document details the objective, technical challenges, and sequential experimental roadmap for the bi-manual garment folding task using the **SmolVLA** architecture.

## 1. Task Description: Bi-Manual Multi-Garment Folding

The objective of the challenge is to develop a robust robotic policy capable of folding a diverse variety of garments with high precision and consistency. 

### **The Robotic System**
- **Platform:** A dual-arm (bi-manual) robot.
- **Action Space:** 12-DOF (6-DOF per arm + 2-DOF for grippers).
- **Control Rate:** 5-10 Hz.

### **The Environment & Perception**
- **Simulation:** Bedroom environment with randomized lighting and textures to simulate real-world variability.
- **Sensors:** Three RGB cameras providing static views of the workspace:
    - **Top View:** Global context of the garment.
    - **Left/Right Views:** Close-up views for precision grasping and sleeve manipulation.
- **Proprioception:** 12-dimensional joint state of the robot arms.

### **The Challenge**
Folding garments is a benchmark problem in deformable object manipulation due to:
- **Cloth Physics:** Non-linear deformation and complex friction.
- **Occlusions:** Fabrics often hide themselves or the robot's grippers during complex folds (e.g., sleeves).
- **Geometric Diversity:** Significant variations in shape, texture, and size between "Pants" and "Tops."

---

## 2. Experimental Timeline & Configurations

We have conducted three distinct training phases to evolve the model from a basic baseline to a more robust, fully-unfrozen architecture.

### **Phase 1: Baseline SmolVLA (Expert-Only)**
The initial strategy focused on utilizing the pre-trained features of the SmolVLM vision backbone while training a lightweight bridge to robotic actions.

- **Configuration:**
    - **Backbone:** Frozen SmolVLM2 (500M).
    - **Action Head:** Trained MLP "Expert."
    - **Training:** 60,000 steps with Batch Size 16.
    - **Augmentation:** None.
- **Performance:** 
    - **Pant Short:** 73.33%
    - **Top Long:** 56.67%
    - **Pant Long:** 35.00%
    - **Top Short:** 11.67%
- **Takeaway:** The model achieved basic grasping but lacked the visual flexibility to handle even minor shifts in garment pose.

### **Phase 2: Visual IQ (Augmentation Run)**
To address the brittleness of Phase 1, we introduced heavy visual noise to force the model to learn geometry over color.

- **Configuration:**
    - **Augmentations:** Color Jitter (saturation up to 1.5) and Random Affine (±5° rotations, ±5% shifts).
    - **Training:** 30,000 steps with Batch Size 64.
- **Performance:**
    - **Pant Short:** **81.67%** (Significant Improvement)
    - **Top Short:** 20.00%
    - **Top Long:** 55.00%
    - **Pant Long:** 30.00%
- **Takeaway:** Augmentations successfully boosted the "easiest" tasks (Short Pants) by making the model more robust to background distractions.

### **Phase 3: Heavy Fine-Tuning (PEFT V2)**
The final phase aimed at unlocking the model’s internal reasoning by allowing it to modify its own internal representation of the vision-action connection.

- **Configuration:**
    - **Framework:** PEFT (LoRA) with Rank 16.
    - **Unfrozen Modules:** Action Head + Vision-to-VLM Connector.
    - **Training:** 40,000 steps.
- **Performance:**
    - **Top Long:** **65.00%** (Record High)
    - **Top Short:** **28.33%**
    - **Pant Long:** **43.00%**
    - **Pant Short:** 78.33%
- **Takeaway:** By unfreezing the connector, the model significantly improved its performance on complex "Long" garments, where spatial coordination between the vision backbone and the robot's hands is most critical.

---

### **Consolidated Success Metrics**

| Garment Type | Phase 1 (Base) | Phase 2 (Aug) | Phase 3 (PEFT V2) |
| :--- | :--- | :--- | :--- |
| **Top Long** | 56.67% | 55.00% | **65.00%** |
| **Top Short** | 11.67% | 20.00% | **28.33%** |
| **Pant Long** | 35.00% | 30.00% | **43.00%** |
| **Pant Short** | 73.33% | **81.67%** | 78.33% |

---
*Last Updated: April 12, 2026*