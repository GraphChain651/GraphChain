export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_FILE="graphchain_for_ppo"
export CUDA_VISIBLE_DEVICES="0,1"

export MODEL_TYPE="qwen2"
export MODEL_PATH="model_saved/qwen2/checkpoint"

python train/RL/ppo/graphchain_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --data_file ${TRAIN_PATH}/${MODEL_TYPE}/${TRAIN_SET}.csv \
    --epochs 5

