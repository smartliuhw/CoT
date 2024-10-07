#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export DS_SKIP_CUDA_CHECK=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

TIME=$(date "+%Y-%m-%d-%H-%M-%S")

TRAIN_DATA="2000-universal_instruct|1000-nq_open_with_snippets|1000-trivia_qa_with_snippets|1000-hotpot_qa_with_snippets"
FORMATTED_TRAIN_DATA=$(echo ${TRAIN_DATA} | sed 's/|/_/g')

MODEL_TYPE=Llama3-8B
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
MAX_LENGTH=1024

torchrun --nproc_per_node=10 ../src/train.py \
    --model_type ${MODEL_TYPE} \
    --train_data $TRAIN_DATA \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --learning_rate ${LR} \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --warmup_steps 20 \
    --report_to wandb \
    --logging_dir ${LOG_DIR} \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --save_safetensors \
    --deepspeed ../src/deepspeed_config.json \
    --seed 725 \
    --bf16 \
    --do_train \
    --save_only_model \
    --max_seq_length ${MAX_LENGTH}
