#!/bin/bash
# Continual Pre-Training (CPT) on Qwen2.5-VL
# This trains on pure text (no images) to adapt the model to a new domain

MODEL_NAME="/mnt/d/working_dir/model_playground/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=8
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Adjust these paths to your data
TRAIN_DATA="./data/ostracod_foundation_cpt.json"

deepspeed src/train/train_cpt.py \
    --deepspeed scripts/zero2_offload.json \
    --model_id $MODEL_NAME \
    --data_path $TRAIN_DATA \
    --freeze_llm False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen_cpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 3.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --max_seq_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard

# Training notes:
# - freeze_llm=False: Train the LLM (main goal of CPT)
# - freeze_vision_tower=True: Keep vision encoder frozen (text-only CPT)
# - freeze_merger=True: Keep vision-language connector frozen
# - learning_rate=1e-5: Lower than SFT (continual pre-training)
# - num_train_epochs=1: Usually 1 epoch is enough for CPT
# - batch_size=4 x 8 = 32 effective: Adjust based on your GPU memory
