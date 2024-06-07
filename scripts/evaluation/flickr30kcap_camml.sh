#!/bin/bash

model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CHUNKS=8

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_cap_loader_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/flickr30k/llava_flickr30k_test.jsonl \
    --image-folder ./data/eval/flickr30k/flickr30k-images \
    --temperature 0 \
    --answers-file ./data/eval/flickr30k/answers/${model:0:7}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --icl_num ${iclnum} \
    --memory ${memory} \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait

output_file=./data/eval/flickr30k/answers/${model:0:7}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/flickr30k/answers/${model:0:7}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


cd data/eval/flickr30k
python convert_jsonl_to_json.py --jsonl answers/${model:0:7}.jsonl --json answers/${model:0:7}.json
#
#
python coco_eval.py answers/${model:0:7}.json test_caption_flickr30k.json


