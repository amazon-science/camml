#!/bin/bash


CHUNKS=8

model=${1}
modelname=./checkpoints/${model}
answerfile=test
iclnum=${2:-"3"}

for IDX in {0..7}; do
    a=$(expr $IDX + 0)
    a="$a"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$a python -m camml.eval.model_vqa_science_ft_camml \
        --model-path $modelname \
        --question-file ./data/scienceqa/llava_test_QCM-LEPA.json \
        --image-folder ./data/scienceqa/images/test \
        --answers-file ./vqa/results/sqa/${answerfile}-chunk$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --icl-num ${iclnum} \
        --conv-mode vicuna_v1 &
done

wait

output_file="./vqa/results/sqa/test.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for idx in $(seq 0 $((CHUNKS-1))); do
  cat "./vqa/results/sqa/${answerfile}-chunk${idx}.jsonl" >> "$output_file"
done

python camml/eval/eval_science_qa_new.py \
    --base-dir ./data/scienceqa \
    --result-file ./vqa/results/sqa/test.jsonl \
    --output-file ./vqa/results/sqa/test_output.json \
    --output-result ./vqa/results/sqa/test_result.json



