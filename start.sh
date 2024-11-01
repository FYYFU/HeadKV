CUDA_VISIBLE_DEVICES='7' python retrieval_head_detection_r2.py  \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --model_provider LLAMA3 \
    --s 0 \
    --e 10000 \
    --task retrieval_reasoning_heads \
    --haystack_dir ./haystack_for_detect_r2


CUDA_VISIBLE_DEVICES='7' python retrieval_head_detection.py  \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --model_provider LLAMA3 \
    --s 0 \
    --e 10000 \
