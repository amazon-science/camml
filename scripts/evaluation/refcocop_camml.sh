#!/bin/bash

model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CHUNKS=8

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$IDX python -m camml.eval.model_refcoco_loader_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/refcocop/llava_refcocop_val.jsonl \
    --image-folder ./data/eval/refcocop/ \
    --temperature 0 \
    --answers-file ./data/eval/refcocop/answers/${model:0:7}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --icl_num ${iclnum} \
    --memory ${memory} \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait

output_file=./data/eval/refcocop/answers/${model:0:7}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/refcocop/answers/${model:0:7}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd data/eval/refcocop
python convert_jsonl_to_json.py --jsonl answers/${model:0:7}.jsonl --json answers/${model:0:7}.json
#
#
python refcoco_eval.py --prediction answers/${model:0:7}.json --gt llava_refcocop_val_gt.json
