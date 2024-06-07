#!/bin/bash


model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
SPLIT="mmbench_dev_cn_20231003"
CKPT=${4}

CHUNKS=8
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_mmbench_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./data/eval/mmbench_cn/answers/$SPLIT/${CKPT}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --lang cn \
    --single-pred-prompt \
    --icl_num ${iclnum} \
    --memory ${memory} \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done


wait

output_file=./data/eval/mmbench_cn/answers/$SPLIT/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/mmbench_cn/answers/$SPLIT/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p data/eval/mmbench_cn/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment ${CKPT}
