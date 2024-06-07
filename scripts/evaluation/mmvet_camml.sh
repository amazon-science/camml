#!/bin/bash

model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CHUNKS=8
CKPT=${4}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./data/eval/mm-vet/images \
    --answers-file ./data/eval/mm-vet/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --icl_num ${iclnum} \
    --memory ${memory} \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

output_file=./data/eval/mm-vet/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/mm-vet/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./data/eval/mm-vet/answers/${CKPT}.jsonl \
    --dst ./data/eval/mm-vet/results/${CKPT}.json

