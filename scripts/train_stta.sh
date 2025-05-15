#!/bin/bash

MODEL_NAME="models/Qwen2.5-7B-Instruct"
FEATURE_DIM=10
HIDDEN_DIMS="256 128 64"
NUM_HEADS=4
DROPOUT=0.1
BATCH_SIZE=8
LEARNING_RATE=0.001 
EPOCHS=10
SAVE_DIR="./checkpoints"
SAVE_INTERVAL=2
USE_CUDA="--use_cuda"
CSV_FILE="qa_data.csv"
GEXF_FILE="graph.gexf"

mkdir -p ${SAVE_DIR}


python train/RL/ppo/train_stta.py \
    --model_name ${MODEL_NAME} \
    --feature_dim ${FEATURE_DIM} \
    --hidden_dims ${HIDDEN_DIMS} \
    --num_heads ${NUM_HEADS} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --save_dir ${SAVE_DIR} \
    --save_interval ${SAVE_INTERVAL} \
    ${USE_CUDA} \
    --csv_file "${CSV_FILE}" \
    --gexf_file "${GEXF_FILE}"

