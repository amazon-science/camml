#!/bin/bash

GPUS_PER_NODE=8


torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$RANDOM \
    camml/train/train_camml_sqa.py \
    --deepspeed "scripts/zero3.json" \
    --model_name_or_path ./checkpoints/vicuna-13b-v1.3 \
    --version v1 \
    --data_path ./data/scienceqa/llava_train_QCM-LEPA.json \
    --image_folder ./data/scienceqa/images/ \
    --vision_tower ./checkpoints/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-pretrain-vicuna-13b-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --perceiver_hidden_size 4608 \
    --perceiver_querys 128 \
    --perceiver_layers 2 \
    --icl_num 3 \
    --random_shots_training True \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir ./checkpoints/camml_13b_sqa_ft \
    --num_train_epochs 12 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True
