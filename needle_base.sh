export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support AdativeKV, ReasonKV
max_capacity_prompts=$3 # 128,2048 in paper
attn_implementation=$4 # Support "flash_attention_2"
model_path=$5
head_choice=$6
beta=$7
temp=$8
s_len=${9}
e_len=${10}
step=${11}
model_provider=${12}
model_version=${13}

save_dir="./needle_results/results_needle_${model_version}_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}" # path to result save_dir

python3 run_needle_in_haystack.py \
    --s_len ${s_len} \
    --e_len ${e_len} \
    --method ${method} \
    --model_path ${model_path} \
    --model_version ${model_version} \
    --model_provider ${model_provider} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --head_choice ${head_choice} \
    --beta ${beta} \
    --temp ${temp} \
    --step ${step} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True