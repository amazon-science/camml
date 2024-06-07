#!/bin/bash


CHUNKS=8
echo $CHUNKS

model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CKPT=${4}
#
#CHUNKS=8
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$IDX python -m camml.eval.model_vqa_science_camml \
      --model-path checkpoints/${model} \
      --question-file ./data/eval/scienceqa/llava_test_CQM-A.json \
      --image-folder ./data/eval/scienceqa/images/test \
      --answers-file ./data/eval/scienceqa/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl \
      --single-pred-prompt \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --memory ${memory} \
      --icl_num ${iclnum} \
      --temperature 0 \
      --conv-mode vicuna_v1 &
done
#
wait
output_file=./data/eval/scienceqa/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/scienceqa/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_science_qa_15.py \
    --base-dir ./data/eval/scienceqa \
    --result-file ./data/eval/scienceqa/answers/${CKPT}.jsonl \
    --output-file ./data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./data/eval/scienceqa/answers/${CKPT}_result.json
