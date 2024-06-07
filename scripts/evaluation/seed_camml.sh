#!/bin/bash

#gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
#IFS=',' read -ra GPULIST <<< "$gpu_list"


model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CHUNKS=8

CKPT=${4}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_loader_camml \
        --model-path checkpoints/${model} \
        --question-file ./data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder ./data/eval/seed_bench \
        --answers-file ./data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --icl_num ${iclnum} \
        --chunk-idx $IDX \
        --memory ${memory} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./data/eval/seed_bench/answers_upload/${model:0:7}.jsonl

