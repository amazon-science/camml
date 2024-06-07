#!/bin/bash



CHUNKS=8


model=${1}
iclnum=${2:-3}
memory=${3:-"665k"}
CKPT=${4}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${IDX} python -m camml.eval.model_vqa_loader_camml \
    --model-path checkpoints/${model} \
    --question-file ./data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./data/eval/textvqa/train_images \
    --answers-file ./data/eval/textvqa/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --memory ${memory} \
    --icl_num ${iclnum} \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done


wait

output_file=./data/eval/textvqa/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/textvqa/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_textvqa \
    --annotation-file ./data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./data/eval/textvqa/answers/${CKPT}.jsonl

