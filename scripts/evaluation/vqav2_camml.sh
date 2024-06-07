#!/bin/bash

CHUNKS=8
model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
SPLIT="llava_vqav2_mscoco_test-dev2015"

CKPT=${4}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_loader_camml \
        --model-path checkpoints/${model} \
        --question-file ./data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./data/eval/vqav2/test2015 \
        --answers-file ./data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --icl_num ${iclnum} \
        --memory ${memory} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
#
wait

output_file=./data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

