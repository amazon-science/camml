#!/bin/bash

model=${1}

CHUNKS=8

iclnum=${2:-3}
memory=${3:-"665k"}
CKPT=${4}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_loader_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./data/eval/vizwiz/test \
    --answers-file ./data/eval/vizwiz/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --memory ${memory} \
    --icl_num ${iclnum} \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

output_file=./data/eval/vizwiz/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/vizwiz/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./data/eval/vizwiz/llava_test.jsonl \
    --result-file ./data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ./data/eval/vizwiz/answers_upload/${CKPT}.json
