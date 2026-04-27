#!/bin/bash

# ACT chunk_size Hyperparameter Sweep Script
# This script runs training sequentially for different chunk_size values.

# Define the values for chunk_size to sweep
# We'll compare a smaller (64) and a larger (128) horizon.
CHUNK_SIZES=(64 128)

# Number of steps per training run (default: 100k)
# You can override this by passing it as the first argument to the script
STEPS=${1:-100000}

# Path to the base config
CONFIG_PATH="configs/train_act_aug_depth.yaml"

# WandB Project Name
WANDB_PROJECT="lehome_act_chunk_sweep"

echo "===================================================="
echo "Starting ACT chunk_size hyperparameter sweep"
echo "Values: ${CHUNK_SIZES[*]}"
echo "Steps per run: $STEPS"
echo "===================================================="

for CHUNK in "${CHUNK_SIZES[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "TIME: $(date)"
    echo "TRAINING: chunk_size=$CHUNK"
    echo "----------------------------------------------------"
    
    # Define unique output directory
    OUTPUT_DIR="outputs/train/act_sweep_chunk_$CHUNK"
    
    # Run lerobot-train with overrides
    # We set n_action_steps = chunk_size as per standard ACT practice
    # We also override steps and wandb settings
    lerobot-train \
        --config_path=$CONFIG_PATH \
        --policy.chunk_size=$CHUNK \
        --policy.n_action_steps=$CHUNK \
        --output_dir=$OUTPUT_DIR \
        --steps=$STEPS \
        --wandb.enable=true \
        --wandb.project=$WANDB_PROJECT \
        --wandb.notes="Sequential sweep run with chunk_size=$CHUNK"
        
    if [ $? -eq 0 ]; then
        echo "Successfully finished training for chunk_size=$CHUNK"
    else
        echo "Error: Training failed for chunk_size=$CHUNK. Moving to next..."
    fi
done

echo "===================================================="
echo "Hyperparameter sweep completed!"
echo "===================================================="
