#!/bin/bash

CHUNKS=8

model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CKPT=${4}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_loader_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./data/eval/pope/val2014 \
    --answers-file ./data/eval/pope/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --icl_num ${iclnum} \
    --memory ${memory} \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done


wait

output_file=./data/eval/pope/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/pope/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_pope.py \
    --annotation-dir ./data/eval/pope/coco \
    --question-file ./data/eval/pope/llava_pope_test.jsonl \
    --result-file ./data/eval/pope/answers/${CKPT}.jsonl
