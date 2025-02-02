#!/bin/bash

# List of model directories
model_dirs=(
    # "../trained/2_1B-prefix-new/checkpoint-last"
    "../trained/2_7B-prefix-new/checkpoint-last"
    "../trained/6B-prefix-new/checkpoint-last"
    "../trained/25_7B-prefix-new/checkpoint-last"
    "../trained/codet5-prefix-new/checkpoint-last"
    "../trained/crystal-prefix-new/checkpoint-last"
)

# Loop through all models
for model_dir in "${model_dirs[@]}"
do
    # Extract the model name (last part of the directory)
    model_name=$(basename $(dirname "$model_dir"))

    # Run sec_eval.py with vLLM
    python sec_eval.py \
        --model_type prefix \
        --model_dir "$model_dir" \
        --output_name "sec-eval-${model_name}" \
        --use_vllm \
        --tensor_parallel_size 4 \
        --max_batch_size 32

    # Sleep to allow garbage collection and free memory
    echo "Sleeping for 10 seconds to allow GC..."
    sleep 10

    echo "Completed evaluation for model: $model_name"
done

echo "All models processed."
