# LeHome Challenge: Training & Evaluation Summary

## 1. Training
- **Config used:** configs/train_dp_all_with_depth.yaml
- **Dataset:** Datasets/example/four_types_merged_with_depth (265,798 frames, 1,000 episodes)
- **Policy:** Diffusion Policy (LeRobot framework)
- **Key hyperparameters:**
  - batch_size: 8
  - steps: 330,000 (≈10 epochs)
  - save_freq: 33,000
  - log_freq: 3,300
- **Output directory:** outputs/train/dp_all_with_depth
- **Training observations:**
  - Loss plateaued after ~1 epoch; early stopping is recommended if loss stops improving.
  - Checkpoints are saved in the output directory; use the 'last' checkpoint for evaluation.

## 2. Evaluation
- **Main script:** scripts/eval.py
- **Policy type:** lerobot
- **Evaluation dataset:** Datasets/example/four_types_merged_with_depth
- **Garment types:** top_long, top_short, pant_long, pant_short
- **Key arguments:**
  - --policy_type lerobot
  - --policy_path outputs/train/dp_all_with_depth/checkpoints/last/pretrained_model
  - --dataset_root Datasets/example/four_types_merged_with_depth
  - --garment_type <type>
  - --num_episodes 5
  - --num_envs <N> (parallel environments, e.g., 2 or 5)
  - --enable_cameras (required for visual policies and video recording)
  - --save_video --video_dir outputs/eval_videos_depth/<type>
  - --headless --device cpu
- **Example command:**

```bash
nohup python -m scripts.eval \
  --policy_type lerobot \
  --policy_path outputs/train/dp_all_with_depth/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/four_types_merged_with_depth \
  --garment_type top_long \
  --num_episodes 5 \
  --num_envs 2 \
  --enable_cameras \
  --save_video \
  --video_dir outputs/eval_videos_depth/top_long \
  --headless \
  --device cpu \
  > logs/eval_depth_baselines/top_long.log 2>&1 &
```

- **Parallelism:** Increasing --num_envs increases CPU usage, not VRAM (simulation is CPU-based).
- **CPU usage:** 100% = 1 core; 200% = 2 cores, etc. Use `htop` or `top` to monitor.
- **Video saving:** Ensure --video_dir is unique per garment/type to avoid overwriting.
- **Log saving:** Use shell redirection (> log.txt 2>&1) to save logs.

## 3. Best Practices & Notes
- Use the 'last' checkpoint for evaluation (lowest loss, most recent state).
- If loss plateaus, consider early stopping.
- For each garment_type, run evaluation separately, but you can use the same merged dataset as dataset_root.
- Monitor CPU and RAM usage when increasing --num_envs.
- If you want structured metrics (CSV/JSON), consider patching the evaluation script to save all_episode_metrics.

## 4. References
- docs/policy_eval.md: Policy evaluation guide
- docs/training.md: Training guide
- docs/datasets.md: Dataset structure and usage
- README.md: Parameter descriptions and quick start

---
This summary provides all context needed for a new coding agent to assist with further LeHome Challenge development, training, or evaluation tasks.