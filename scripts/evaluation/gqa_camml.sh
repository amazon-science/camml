#!/bin/bash

CHUNKS=8
model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}

CKPT=${4}
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_loader_camml \
        --model-path checkpoints/${model} \
        --question-file ./data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./data/eval/gqa/data/images \
        --answers-file ./data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --icl_num ${iclnum} \
        --memory ${memory} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
#
output_file=./data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl
#
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/${model:0:7}_testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced --expname ${model:0:7}
