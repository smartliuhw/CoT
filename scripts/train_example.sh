#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export DS_SKIP_CUDA_CHECK=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

TIME=$(date "+%Y-%m-%d-%H-%M-%S")

TRAIN_DATA="10000-universal_instruct|500-nq_open|500-trivia_qa|500-hotpot_qa"
FORMATTED_TRAIN_DATA=$(echo ${TRAIN_DATA} | sed 's/|/_/g')

MODEL_TYPE=Gemma-7B
LR=5e-5

MODEL_NAME=${MODEL_TYPE}_${TIME}_${LR}_${FORMATTED_TRAIN_DATA}
OUTPUT_DIR=/home/shared_space/smart/cot_train/output/${MODEL_NAME}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi
LOG_DIR=/home/shared_space/smart/cot_train/tensorboard_logs/${MODEL_NAME}
if [ ! -d ${LOG_DIR} ]; then
    mkdir -p ${LOG_DIR}
fi
MAX_LENGTH=1600

torchrun --nproc_per_node=10 ../src/train.py \
    --model_type ${MODEL_TYPE} \
    --train_data $TRAIN_DATA \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --learning_rate ${LR} \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --logging_dir ${LOG_DIR} \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 5 \
    --save_safetensors \
    --deepspeed ../src/deepspeed_config.json \
    --seed 725 \
    --bf16 \
    --do_train \
    --max_seq_length ${MAX_LENGTH}
